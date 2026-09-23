import math
import string

class SemanticHistoryRetriever:
    """
    Advanced context history retriever using an Inverted Index, TF-IDF,
    and Cosine Similarity for O(1)/O(W) search complexity.

    Optimizes:
      - Space Complexity: Keeps the LLM context window small by retrieving only
        historically relevant turns instead of sending the entire conversation history.
      - Time Complexity: Performs microsecond keyword-based lookup rather than O(N)
        brute force scans.
    """
    def __init__(self, max_history=50):
        self.history = []  # List of conversation turns: [{'role': str, 'content': str}]
        self.inverted_index = {}  # Inverted index: word -> list of (turn_index, normalized_tf)
        self.df = {}  # Document Frequency: word -> count of user turns containing the word
        self.max_history = max_history  # Maximum number of turns to keep in history

        # Stopwords to filter out grammatical noise terms
        self.stopwords = {
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
            "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself",
            "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
            "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be",
            "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
            "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by",
            "for", "with", "about", "against", "between", "into", "through", "during", "before",
            "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
            "under", "again", "further", "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
            "will", "just", "don", "should", "now", "yeah", "yes", "true", "ok", "okay"
        }

    def _tokenize(self, text):
        """ Tokenizes text by lowercasing, removing punctuation, and filtering stopwords. """
        text = text.lower()
        # Strip punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        words = text.split()
        return [w for w in words if w not in self.stopwords]

    def _rebuild_indices(self):
        """Rebuild the inverted index and document frequency from scratch.
        Called when history is truncated to maintain index consistency."""
        # Reset indices
        self.inverted_index = {}
        self.df = {}

        # Re-process all user turns in history
        for turn_index, turn in enumerate(self.history):
            if turn['role'] == 'user':
                words = self._tokenize(turn['content'])
                if not words:
                    continue

                # Calculate term frequencies
                tf = {}
                for w in words:
                    tf[w] = tf.get(w, 0) + 1

                # Normalize term frequencies by max frequency in this document
                max_tf = max(tf.values()) if tf else 1
                for w, count in tf.items():
                    normalized_tf = count / max_tf

                    # Update inverted index postings list
                    if w not in self.inverted_index:
                        self.inverted_index[w] = []
                        self.df[w] = 0
                    self.inverted_index[w].append((turn_index, normalized_tf))

                # Update Document Frequency (DF)
                for w in set(words):
                    self.df[w] = self.df.get(w, 0) + 1

    def add_turn(self, role, content):
        """ Adds a conversation turn and updates the TF-IDF index. """
        turn_index = len(self.history)
        self.history.append({'role': role, 'content': content})

        # Enforce maximum history length
        if len(self.history) > self.max_history:
            # Remove oldest turns to maintain limit
            excess = len(self.history) - self.max_history
            self.history = self.history[excess:]
            # Rebuild indices since turn indices have changed
            self._rebuild_indices()
            return  # Indices already rebuilt, skip normal indexing

        # We only index user queries for similarity lookups
        if role == 'user':
            words = self._tokenize(content)
            if not words:
                return

            # Calculate term frequencies
            tf = {}
            for w in words:
                tf[w] = tf.get(w, 0) + 1

            # Normalize term frequencies by max frequency in this document
            max_tf = max(tf.values()) if tf else 1
            for w, count in tf.items():
                normalized_tf = count / max_tf

                # Update inverted index postings list
                if w not in self.inverted_index:
                    self.inverted_index[w] = []
                    self.df[w] = 0
                self.inverted_index[w].append((turn_index, normalized_tf))

            # Update Document Frequency (DF)
            for w in set(words):
                self.df[w] = self.df.get(w, 0) + 1

    def retrieve_relevant_context(self, query, top_k=2):
        """
        Retrieves the indices of the top-K most semantically matching user turns 
        using Cosine Similarity of TF-IDF vectors.
        """
        query_words = self._tokenize(query)
        if not query_words or not self.history:
            return []
            
        # Calculate TF for query
        query_tf = {}
        for w in query_words:
            query_tf[w] = query_tf.get(w, 0) + 1
            
        total_user_docs = len([h for h in self.history if h['role'] == 'user'])
        if total_user_docs == 0:
            return []
            
        # Compute query vector TF-IDF values
        query_vector = {}
        max_q_tf = max(query_tf.values())
        for w, count in query_tf.items():
            df_val = self.df.get(w, 0)
            # IDF log formulation
            idf = math.log(1.0 + (total_user_docs / df_val)) if df_val > 0 else 0.0
            query_vector[w] = (count / max_q_tf) * idf

        # Calculate scores and document vector magnitudes
        scores = {}  # turn_index -> dot_product
        doc_lengths = {}  # turn_index -> sum of squared TF-IDFs
        
        for w, q_tfidf in query_vector.items():
            if q_tfidf == 0:
                continue
            postings = self.inverted_index.get(w, [])
            df_val = self.df.get(w, 0)
            idf = math.log(1.0 + (total_user_docs / df_val)) if df_val > 0 else 0.0
            
            for turn_index, doc_tf in postings:
                doc_tfidf = doc_tf * idf
                scores[turn_index] = scores.get(turn_index, 0.0) + (q_tfidf * doc_tfidf)
                doc_lengths[turn_index] = doc_lengths.get(turn_index, 0.0) + (doc_tfidf * doc_tfidf)

        # Normalize to calculate Cosine Similarity
        query_length = math.sqrt(sum(v*v for v in query_vector.values()))
        if query_length == 0:
            return []
            
        cosine_similarities = []
        for turn_index, dot_product in scores.items():
            doc_len = math.sqrt(doc_lengths[turn_index])
            if doc_len > 0:
                similarity = dot_product / (query_length * doc_len)
            else:
                similarity = 0.0
            cosine_similarities.append((turn_index, similarity))

        # Sort matches by score descending
        cosine_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return indices with a positive match score above threshold (0.15)
        relevant_turns = [turn_idx for turn_idx, sim in cosine_similarities if sim > 0.15]
        return relevant_turns[:top_k]
