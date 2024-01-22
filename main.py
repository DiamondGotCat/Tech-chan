from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import json

nltk.download('punkt')
nltk.download('stopwords')

class ChatBot:
    def __init__(self):
        self.responses = {}
        self.emotion_model = None

    def preprocess_text(self, text):
        # 英語のテキスト処理
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalpha()]
        tokens = [token for token in tokens if token not in stopwords.words('english')]
        return ' '.join(tokens)

    def train_emotion_model(self, data):
        # 学習データとラベルの取得
        texts = [self.preprocess_text(item) for item in data.keys()]
        labels = [data[item]["emotion_level"] for item in data.keys()]

        # テキストデータをベクトル化
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(texts)

        # 手動で学習データとテストデータに分割 (80%学習データ, 20%テストデータ)
        split_index = int(0.8 * len(labels))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = labels[:split_index], labels[split_index:]

        # Naive Bayesモデルのトレーニング
        self.emotion_model = MultinomialNB()
        self.emotion_model.fit(X_train, y_train)

        # テストデータでの評価
        accuracy = self.emotion_model.score(X_test, y_test)
        print(f"Emotion Model Accuracy: {accuracy}")

    def learn_response(self, user_input):
        # 学習データ内にない場合の処理
        response = input("Enter an appropriate response: ")
        emotion_level = int(input("Rate the emotion level of the response from 1 to 10: "))
        
        # 学習データに追加
        self.responses[user_input] = {"response": response, "emotion_level": emotion_level}

        # JSONファイルに保存
        with open('responses.json', 'w') as f:
            json.dump(self.responses, f, ensure_ascii=False, indent=4)

    def get_emotion_level(self, text):
        # 入力テキストをベクトル化
        processed_text = self.preprocess_text(text)
        vectorizer = CountVectorizer()
        X = vectorizer.transform([processed_text])

        # 学習済み感情モデルを使用して感情レベルを予測
        emotion_level = self.emotion_model.predict(X)[0]
        return emotion_level

    def chat(self, user_input):
        processed_input = self.preprocess_text(user_input)

        # 学習データ内にある場合はそのまま回答
        if processed_input in self.responses:
            return self.responses[processed_input]["response"]

        # 学習データ内にない場合の例外処理
        else:
            print("This is an unseen question. Let's learn!")
            self.learn_response(processed_input)
            return "I have learned. Please ask again."

# チャットボットの初期化
chatbot = ChatBot()

# チャットの開始
while True:
    user_input = input("User: ")
    if user_input.lower() == 'exit':
        break
    response = chatbot.chat(user_input)
    print(f"ChatBot: {response}")
