import os

from flask import Flask, request, abort

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent,
    TextMessage,
    TextSendMessage,
)

from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

from character_setting import character_setting

app = Flask(__name__)

# LINE APIの準備
line_bot_api = LineBotApi(os.environ["CHANNEL_ACCESS_TOKEN"])
handler = WebhookHandler(os.environ["CHANNEL_SECRET"])

# 会話履歴ストア
store = {}


# セッションIDごとの会話履歴の取得
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# チャットプロンプトテンプレート
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=character_setting),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)

# チャットモデル
llm = ChatOpenAI(
    model_name="gpt-4o",
    max_tokens=512,
    temperature=0.2,
    streaming=True,
)

# パース用モジュール(レスポンスのJSONからcontentを取り出すパーサー)
parser = StrOutputParser()

# LCEL
runnable = prompt | llm | parser

# RunnableWithMessageHistoryでラップ
runnable_with_history = RunnableWithMessageHistory(
    runnable=runnable,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


@app.route("/")
def hello_world():
    return "It Works!"


@app.route("/callback", methods=["POST"])
def callback():
    # get X-Line-Signature header value
    signature = request.headers["X-Line-Signature"]

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print(
            "Invalid signature. Please check your channel access token/channel secret."
        )
        abort(400)

    return "OK"


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    if event.message.text == "リセット":
        store.clear()
        response = "会話をリセットしました。"
    else:
        response = runnable_with_history.invoke(
            {"input": event.message.text},
            config={"configurable": {"session_id": "hoge"}},
        )
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=response))


if __name__ == "__main__":
    app.run()
