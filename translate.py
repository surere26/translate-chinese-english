import sys
import asyncio
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from threading import Thread
import time

app = Flask(__name__)
socketio = SocketIO(app)

# 模型加载状态和翻译管道
model_loaded = False
zh2en_pipeline = None
en2zh_pipeline = None


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    global model_loaded
    if model_loaded:
        emit('model_status', {'status': 'loaded'})
    else:
        emit('model_status', {'status': 'loading'})


@socketio.on('translate')
def handle_translate(data):
    global model_loaded, zh2en_pipeline, en2zh_pipeline

    if not model_loaded:
        emit('translation_result', {'error': '模型正在加载中，请稍候...'})
        return

    text = data.get('text', '')
    if not text:
        emit('translation_result', {'error': '请输入要翻译的文本'})
        return

    try:
        # 检测语言并进行翻译
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            result = zh2en_pipeline(input=text)
            emit('translation_result', {'result': '中文→英文：' + result['translation']})
        else:
            result = en2zh_pipeline(input=text)
            emit('translation_result', {'result': '英文→中文：' + result['translation']})
    except Exception as e:
        emit('translation_result', {'error': f'翻译出错：{str(e)}'})


def load_models():
    global model_loaded, zh2en_pipeline, en2zh_pipeline

    try:
        # 加载中译英模型
        zh2en_pipeline = pipeline(task=Tasks.translation, model="iic/nlp_csanmt_translation_zh2en")
        # 加载英译中模型
        en2zh_pipeline = pipeline(task=Tasks.translation, model="iic/nlp_csanmt_translation_en2zh")

        # 预加载测试
        test_text_zh = "测试"
        result_zh = zh2en_pipeline(input=test_text_zh)
        print(f"预加载测试 - 中文→英文：{result_zh['translation']}")

        test_text_en = "detect"
        result_en = en2zh_pipeline(input=test_text_en)
        print(f"预加载测试 - 英文→中文：{result_en['translation']}")

        model_loaded = True
        # 通知所有客户端模型已加载完成
        socketio.emit('model_status', {'status': 'loaded'})
        print("模型预加载完成")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        model_loaded = False
        socketio.emit('model_status', {'status': 'error', 'message': str(e)})


if __name__ == '__main__':
    # 启动模型加载线程
    model_thread = Thread(target=load_models)
    model_thread.daemon = True
    model_thread.start()

    # 启动Flask应用
    print("服务器启动中，访问 http://localhost:8961 使用应用")
    socketio.run(app, host='0.0.0.0', port=8961,allow_unsafe_werkzeug=True)