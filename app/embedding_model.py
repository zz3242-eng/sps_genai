import spacy


class EmbeddingModel:
    """Word embedding model using spaCy, adapted from Module 2's Word Embeddings notebook."""

    def __init__(self, model_name="en_core_web_md"):
        self.nlp = spacy.load(model_name)

    def get_embedding(self, word):
        return self.nlp(word).vector.tolist()

    def similarity(self, word1, word2):
        return self.nlp(word1).similarity(self.nlp(word2))
