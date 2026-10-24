"""Flask-SocketIO server: thin routing layer over `retriever.session`.

All business logic (KV-cache save/restore/clear, conversation state, chat persistence,
index building, folder picking) lives outside this file. Handlers here are 1-5 lines
that delegate.
"""
import os
import threading
import traceback

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from core.config import ADVANCED_RESEARCH, REASONING_MODEL, USER_INDEX_PATHS, ENABLE_LOGGING
from core import cancel
from core.utils import strip_emoji
from components.generators import Generator
from components.indexer.add_w_folder import build_index_from_folder
from .folder_picker import pick_folder
from core.logging_config import get_logger

logger = get_logger(__name__)


socketio = None
_indexing_in_progress = False


def _lifecycle_command(query, sid):
    """Handle 'exit' / 'restart' commands typed in the chat box. Returns True if handled."""
    cmd = query.strip().lower()
    if cmd == 'exit':
        print("\nStopping ALMA (exit typed in the web interface).")
        emit('server_shutdown', {'message': 'Server is shutting down...'}, to=sid)

        def shutdown():
            import time, signal
            time.sleep(0.5)
            os.kill(os.getpid(), signal.SIGINT)
        threading.Thread(target=shutdown, daemon=True).start()
        return True

    if cmd == 'restart':
        if os.name == 'nt':
            # os.execv on Windows spawns a new process and exits the old one, which
            # detaches the server from its console window. Restart by hand instead.
            emit('error', {'message': "Restart is not supported on Windows. "
                                      "Type 'exit', then start the app again."}, to=sid)
            return True
        print("\nRestarting ALMA (restart typed in the web interface)...")
        emit('server_restart', {'message': 'Server is restarting...'}, to=sid)

        def restart():
            import time, stat, sys
            time.sleep(0.5)
            # Close socket FDs to release the port; keep file FDs for exec.
            for fd in range(3, 256):
                try:
                    if stat.S_ISSOCK(os.fstat(fd).st_mode):
                        os.close(fd)
                except OSError:
                    pass
            os.execv(sys.executable, [sys.executable] + sys.argv)
        threading.Thread(target=restart, daemon=True).start()
        return True

    return False


def create_app(retriever, host='0.0.0.0', port=5050):
    global socketio

    app = Flask(__name__, static_folder='static', template_folder='templates')
    CORS(app)
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading',
                        ping_timeout=60, ping_interval=25)

    storage = retriever.session.storage

    # ── Pages ────────────────────────────────────────────────────────
    @app.route('/')
    def index():
        return render_template('standalone.html')

    @app.route('/status')
    def status():
        return {'status': 'ready', 'mode': 'websocket'}

    # ── Chat REST (read-only) ────────────────────────────────────────
    @app.route('/api/chats', methods=['GET'])
    def list_chats():
        return jsonify(storage.list_chats())

    @app.route('/api/chats/<chat_id>', methods=['GET'])
    def get_chat(chat_id):
        chat = storage.load_chat(chat_id)
        return (jsonify(chat) if chat else (jsonify({'error': 'Chat not found'}), 404))

    @app.route('/api/chats/<chat_id>', methods=['DELETE'])
    def delete_chat(chat_id):
        return (jsonify({'success': True}) if retriever.session.delete_chat(chat_id)
                else (jsonify({'error': 'Chat not found'}), 404))

    # ── Index management ─────────────────────────────────────────────
    @app.route('/api/index/status', methods=['GET'])
    def index_status():
        return jsonify({
            'user_index_exists': os.path.exists(USER_INDEX_PATHS['faiss_index_path']),
            'active': 'user' if retriever._index_paths is not None else 'default',
            'indexing_in_progress': _indexing_in_progress,
        })

    @app.route('/api/index/browse', methods=['GET'])
    def browse_folder():
        result = pick_folder()
        status_code = 500 if result.get('error') else 200
        return jsonify(result), status_code

    # ── WebSocket lifecycle ──────────────────────────────────────────
    @socketio.on('connect')
    def handle_connect():
        emit('status', {'state': 'connected', 'message': 'Connected to RAG server'})

    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info("Client disconnected — any ongoing generation will continue but won't emit events")

    @socketio.on('abort_generation')
    def handle_abort_generation(data):
        # Stops at the next token or prefill step; run_cancellable rolls back.
        cancel.request()

    # ── WebSocket chat session ───────────────────────────────────────
    @socketio.on('clear_cache')
    def handle_clear_cache(data):
        try:
            retriever.session.clear()
            emit('status', {'state': 'connected', 'message': 'Conversation cleared - Ready for new conversation'})
        except Exception as e:
            logger.warning(f"Error clearing session: {e}")
            emit('status', {'state': 'connected', 'message': 'Ready'})

    @socketio.on('restore_chat')
    def handle_restore_chat(data):
        chat_id = data.get('chat_id')
        messages = data.get('messages', [])
        try:
            retriever.session.restore(chat_id, fallback_messages=messages)
            emit('status', {'state': 'connected', 'message': 'Conversation restored'})
        except Exception as e:
            logger.warning(f"Error restoring conversation: {e}")
            traceback.print_exc()
            emit('status', {'state': 'connected', 'message': 'Conversation restored (without cache)'})

    @socketio.on('save_chat')
    def handle_save_chat(data):
        chat_id = data.get('chat_id')
        messages = data.get('messages', [])
        title = data.get('title')
        is_new = not chat_id
        if chat_id:
            retriever.session.current_chat_id = chat_id
        chat = retriever.session.save_messages(messages, title)
        if chat:
            emit('chat_saved', {'chat_id': chat['id'], 'chat': chat, 'is_new': is_new})
        else:
            emit('error', {'message': 'Failed to save chat'})

    # ── Index building (folder → user index) ─────────────────────────
    @socketio.on('build_user_index')
    def handle_build_user_index(data):
        global _indexing_in_progress
        folder_path = data.get('folder_path', '').strip()
        sid = request.sid

        if not folder_path or not os.path.isdir(folder_path):
            emit('index_error', {'message': f'Folder not found: {folder_path}' if folder_path else 'No folder path provided.'})
            return
        if _indexing_in_progress:
            emit('index_error', {'message': 'Indexing is already in progress.'})
            return

        _indexing_in_progress = True
        emit('index_progress', {'stage': 'loading', 'percent': 0, 'message': 'Starting indexing...'})

        def relay(stage, percent, message):
            socketio.emit('index_progress', {'stage': stage, 'percent': percent, 'message': message}, to=sid)

        def run():
            global _indexing_in_progress
            try:
                chunk_count = build_index_from_folder(folder_path, USER_INDEX_PATHS, progress_callback=relay)
                retriever.switch_to_user_index(USER_INDEX_PATHS)
                socketio.emit('index_complete', {'message': 'Custom index built and activated.', 'chunks': chunk_count}, to=sid)
            except Exception as e:
                print(f"Indexing the folder failed: {e}")
                logger.warning(traceback.format_exc())
                socketio.emit('index_error', {'message': f'Indexing failed: {str(e)}'}, to=sid)
            finally:
                _indexing_in_progress = False

        threading.Thread(target=run, daemon=True).start()

    @socketio.on('activate_user_index')
    def handle_activate_user_index(data):
        if not os.path.exists(USER_INDEX_PATHS['faiss_index_path']):
            emit('index_error', {'message': 'No user index found. Please build one first.'})
            return
        retriever.switch_to_user_index(USER_INDEX_PATHS)
        emit('index_switched', {'active': 'user', 'message': 'Switched to custom index.'})

    @socketio.on('reset_to_default_index')
    def handle_reset_to_default_index(data):
        retriever.switch_to_default_index()
        emit('index_switched', {'active': 'default', 'message': 'Switched to default index.'})

    # ── Query (main entry point for chat) ────────────────────────────
    @socketio.on('query')
    def handle_query(data):
        query = data.get('query', '')
        message_id = data.get('message_id', 'unknown')
        sid = request.sid

        if not query:
            emit('error', {'message': 'No query provided'})
            return
        if _lifecycle_command(query, sid):
            return

        logger.info(f"\n{'='*60}\nProcessing query: {query}\nMessage ID: {message_id}\n{'='*60}\n")

        def stream_callback(event_type, payload):
            # Fires from the MLX thread (no request context), so use socketio.emit by sid.
            try:
                socketio.emit(event_type, payload, to=sid)
            except Exception as e:
                logger.warning(f"Error emitting {event_type}: {e}")

        try:
            emit('status', {'state': 'thinking', 'message': 'Thinking…'})

            session = retriever.session
            if ADVANCED_RESEARCH or not REASONING_MODEL:
                generator_cls = Generator
            else:
                from components.generators.LRM import LRMGenerator
                generator_cls = LRMGenerator

            # Generation must run on the MLX thread that owns the model/cache (see mlx_runtime).
            try:
                response = cancel.run_cancellable(
                    session,
                    generator_cls.answer_query_with_llm,
                    query=query,
                    llm_model=retriever.llm_model,
                    llm_tokenizer=retriever.llm_tokenizer,
                    retriever=retriever,
                    prompt_cache=session.prompt_cache,
                    stream_callback=stream_callback,
                    conversation_manager=session.conversation_manager,
                )
            except cancel.Cancelled:
                # The partial answer stays in the conversation; save it like any turn.
                emit('done', {'message_id': message_id, 'response': '', 'cancelled': True})
                if session.current_chat_id:
                    session.save_cache()
                return

            if isinstance(response, tuple):
                response_text, last_prompt = response
            else:
                response_text, last_prompt = response, None

            emit('done', {'message_id': message_id, 'response': strip_emoji(response_text or '')})
            logger.info(f"\nResponse complete. (Message ID: {message_id})")

            if session.current_chat_id:
                session.save_cache(last_prompt=last_prompt)

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"\n{error_msg}")
            logger.warning(traceback.format_exc())
            emit('error', {'message': error_msg})
            emit('done', {'message_id': message_id, 'error': True})

    return app, socketio


def run_server(retriever, host='0.0.0.0', port=5050):
    app, sock = create_app(retriever, host, port)
    if not ENABLE_LOGGING:
        import flask.cli
        flask.cli.show_server_banner = lambda *args, **kwargs: None   # " * Serving Flask app ..."
    shown_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    print(f"\nALMA is ready: open http://{shown_host}:{port} in your browser. Press Ctrl+C to stop.\n")
    # allow_unsafe_werkzeug: this is a local research server; newer flask-socketio
    # refuses to start on Werkzeug in non-debug mode without this flag.
    # log_output: Werkzeug's per-request lines only when logging is on.
    sock.run(app, host=host, port=port, debug=False, use_reloader=False,
             allow_unsafe_werkzeug=True, log_output=ENABLE_LOGGING)
