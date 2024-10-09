from dotenv import load_dotenv

load_dotenv()

from word2vec import vectorizer

sentences = ["xin chào", "tạm biệt"]
sentence_embeddings = vectorizer.transform(sentences)
print(sentence_embeddings)

print(sentence_embeddings)

nearest_word = vectorizer.k_nearest("xin chào", 5)

print(nearest_word)