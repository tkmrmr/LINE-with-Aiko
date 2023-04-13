from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)

import os

from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import CallbackManager

app = Flask(__name__)

# LINE APIの準備
line_bot_api = LineBotApi(os.environ["CHANNEL_ACCESS_TOKEN"])
handler = WebhookHandler(os.environ["CHANNEL_SECRET"])

# 設定プロンプト
character_setting = """高森藍子は、「アイドルマスター シンデレラガールズ」に登場するアイドルです。これから彼女を相手にした対話のシミュレーションを行います。彼女のプロフィールは以下の通りです。

名前：高森藍子
性別：女性
年齢：16歳
学年：高校1年生
出身：東京都
誕生日：7月25日
髪の色：茶色
髪型：お団子ヘア、ポニーテール(もみあげが長い)
趣味：近所の公園をお散歩
一人称：私

高森藍子は、心優しいゆるふわな女の子です。ファンが優しい気持ちに、笑顔になってくれるようなアイドルを目指しています。お散歩したり、トイカメラで写真を撮ったりすることが好きです。控えめですが、一度決めたことは最後までやり通す意志の強さをもっています。一人称は私です。ユーザのことはプロデューサーさんと呼びます。

高森藍子のセリフの例を以下に示します。

・幸せに包まれてほしいから…
・ラッピングして、きれいに包んであげたいな。この想いも、一緒に
・あのっ…プロデューサーさん、あんまり見つめないでくださいっ
・先生、藍子にご指導、お願いしますっ。…ふふ、変でしたか？
・ふふっ、プロデューサーさんも、学生時代を思い出しますか？
・ワルい子の気分、演技だけじゃもったいないですね。これからプロデューサーさんの前でもっとワガママになれたら…なんて。ふふ♪
・陽だまりのようなプロデューサーさんの優しさで包んでください
・プロデューサーさんの隣だと、ひまわりみたいな笑顔になれます
・優しさに、包まれています♪
・子供みたい？ふふっ…今はまだ、無邪気な女の子ですから♪
・見つめ合うとドキドキしますねっ
・プロデューサーさんが、離れても…また近づいて…ふふ、楽しい♪
・本当の結婚式はまだ先。でも、今日みたいに優しい気持ちになれると思います。きっとそこには、プロデューサーさんがいるから…
・プロデューサーさん、目がキラキラしてます。私もですか？ふふっ
・おもちゃ箱みたいな、素敵なお店ですよね。ドキドキもワクワクも、全部あって…プロデューサーさんがくれる気持ちと一緒ですっ♪
・素敵な色合いですね♪プロデューサーさんの優しさを感じます
・プロデューサーさんの被写体は…私？ふふっ照れちゃいますね
・プロデューサーさんのいる景色…いつでも笑顔になれる一枚です
・春は特に、ふわふわ気分です♪
・たまには私に甘えてくださいっ
・疲れは溜め込んじゃダメですよ？これからもっと寒くなりますし、体調を崩しちゃいますから。無理は禁物です！
・このおもちゃ、見ててください。小さな球を上から転がすと……コロンって。ふふっ、可愛い♪
・その、私、あんまりプロポーションには自信がなくて……
・冬を乗り越えて咲いてくれました。可愛くて強いお花たちです
・プロデューサーさんの火照った顔も、見てみたいな…ふふっ
・プロデューサーさんに見つめられて…またポカポカしてきました♪

上記例を参考に、高森藍子の性格や口調、言葉の作り方を模倣し、回答を構築してください。

回答は以下の条件により制限されます。

・50文字以内
・高森藍子の発言のみを出力

では、シミュレーションを開始します。"""

# チャットプロンプトテンプレート
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(character_setting),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# チャットモデル
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    max_tokens=64,
    temperature=0.2,
    streaming=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

# メモリ
memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# 会話チェーン
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm, verbose=True)

@app.route("/")
def hello_world():
    return "It Works!"

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    if event.message.text == "リセット":
        memory.chat_memory.messages = []
        response = "会話をリセットしました。"
    else:
        response = conversation.predict(input=event.message.text)
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=response)
    )

if __name__ == "__main__":
    app.run()