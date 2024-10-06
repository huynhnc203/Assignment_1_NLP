from dotenv import load_dotenv

load_dotenv()

from word2vec import vectorizer

sentences = ["xin chào", "tạm biệt"]
sentence_embeddings = vectorizer.transform(sentences)
print(sentence_embeddings)