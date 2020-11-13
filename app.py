
import inline as inline
from flask import Flask,render_template,url_for,request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict() :
	df= pd.read_csv("D:\Desktop\SMS_spam_detector\data.csv", encoding="latin-1")
	df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
	df.columns = ["label", "message"]
	# df.head()

	# Some EDA
	print(df.describe())

	print(df.groupby('label').describe())

	# Let's make a new column to detect how long the text messages are:
	df['length'] = df['message'].apply(len)
	print(df.length.describe())

	# There's one exceptionally long message of length 910
	df[df['length'] == 910]['message'].iloc[0]


	# trying to see if message length is a distinguishing feature between ham and spam:
	plt.figure(figsize=(14, 8))

	df[df.label == 'ham'].length.plot(bins=20, kind='hist', color='blue',
									  label='Ham messages', alpha=0.6)
	df[df.label == 'spam'].length.plot(kind='hist', color='red',
									   label='Spam messages', alpha=0.6)
	plt.legend()
	plt.xlabel("Message Length")
	# Through just basic EDA we've been able to discover a trend that spam messages tend to have more characters.


	df['label']=df['label'].map({'ham': 0, 'spam': 1})
	X = df['message']
	y = df['label']

	# Preprocessing(Removing stopwords, stemming etc)
	from nltk.stem import WordNetLemmatizer
	lemmatizer = WordNetLemmatizer()

	from nltk.corpus import wordnet
	from nltk import pos_tag
	def get_simple_pos(tag):

		if tag.startswith('J'):
			return wordnet.ADJ
		elif tag.startswith('V'):
			return wordnet.VERB
		elif tag.startswith('N'):
			return wordnet.NOUN
		elif tag.startswith('R'):
			return wordnet.ADV
		else:
			return wordnet.NOUN

	from nltk.corpus import stopwords
	import string
	stops = set(stopwords.words('english'))
	punctuations = list(string.punctuation)
	stops.update(punctuations)
	stops, string.punctuation


	def clean_text(message):
		words = word_tokenize(message)
		output_words = []
		for w in words:
			if w.lower() not in stops:
				pos = pos_tag([w])
				clean_word = lemmatizer.lemmatize(w, pos=get_simple_pos(pos[0][1]))
				output_words.append(clean_word.lower())
		return " ".join(output_words)


	from nltk.tokenize import word_tokenize
	cleanX = [clean_text(message) for message in X]


	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(cleanX, y, test_size=0.33, random_state=42)


	# Building the vocabulary and transforming the data from text to numerical format
	from sklearn.feature_extraction.text import CountVectorizer
	count_vec = CountVectorizer(max_features=10000, ngram_range=(1, 2), max_df=0.7)


	X_train = count_vec.fit_transform(X_train)
	count_vec.get_feature_names()


	X_test = count_vec.transform(X_test)


	# The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification).
	from sklearn.naive_bayes import MultinomialNB
	clf = MultinomialNB()
	clf.fit(X_train, y_train)
	predictions = clf.predict(X_test)


	# There are quite a few possible metrics for evaluating model performance. Which one is the most important depends on the task and the business effects of decisions based off of the model. For example, the cost of mis-predicting "spam" as "ham" is probably much lower than mis-predicting "ham" as "spam".
	print(clf.score(X_test, y_test))  # Simple accuracy is not really helpful since our data is imbalanced
	from sklearn.metrics import classification_report
	print(classification_report(y_test, predictions))


	y_pred_prob = clf.predict_proba(X_test)


	# calculate AUROC
	from sklearn import metrics
	print(metrics.roc_auc_score(y_test, y_pred_prob[:, 1]))


	# from sklearn.ensemble import RandomForestClassifier
	# clf = RandomForestClassifier(random_state=0)
	# clf.fit(X_train,y_train)
	# predictions=clf.predict(X_test)
	# clf.score(X_test,y_test)


	# Rather than retraining the model each time an API call is made, we can just save the model and the vectorizer after training and then simply load them when a new  API call is made
	import pickle
	path = "D:\Desktop\SMS_spam_detector"
	model_path = path + "\model.pkl"
	vectorizer_path = path + "\count_vec.pkl"
	pickle.dump(clf, open(model_path, 'wb'))
	pickle.dump(count_vec, open(vectorizer_path, 'wb'))
	# clf = pickle.load(open(model_path, 'rb'))
	# count_vec = pickle.load(open(vectorizer_path, "rb"))

	if request.method == 'POST':
		message = request.form['message']
		data = [clean_text(message)]
		data_dtm = count_vec.transform(data)
		my_prediction = clf.predict(data_dtm)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)

