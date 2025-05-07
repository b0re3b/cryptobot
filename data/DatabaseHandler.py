def save_topic_models(self):
    try:
        # Create directory if it doesn't exist
        os.makedirs(self.topic_model_dir, exist_ok=True)

        # Save vectorizer
        if self.vectorizer:
            joblib.dump(self.vectorizer, os.path.join(self.topic_model_dir, 'vectorizer.pkl'))

        # Save LDA model
        if self.lda_model:
            joblib.dump(self.lda_model, os.path.join(self.topic_model_dir, 'lda_model.pkl'))

        # Save NMF model
        if self.nmf_model:
            joblib.dump(self.nmf_model, os.path.join(self.topic_model_dir, 'nmf_model.pkl'))

        # Save KMeans model
        if self.kmeans_model:
            joblib.dump(self.kmeans_model, os.path.join(self.topic_model_dir, 'kmeans_model.pkl'))

        # Save topic words
        if self.topic_words:
            joblib.dump(self.topic_words, os.path.join(self.topic_model_dir, 'topic_words.pkl'))

        self.logger.info("Topic models saved to disk")
    except Exception as e:
        self.logger.error(f"Error saving topic models: {e}")