# Add these new methods to your existing XBotDetectorDB class in src/storage.py

    def get_tweets_for_analysis(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieves tweets that have been processed (cleaned) but not yet analyzed by our model.
        This is the main input for the batch inference pipeline.
        """
        try:
            # We fetch tweets that Tanay's cleaning script has marked as 'processed'
            # but that our inference pipeline has not yet marked as 'bot_analyzed'.
            cursor = self.tweets_collection.find(
                {"processed": True, "bot_analyzed": False}
            ).limit(limit)
            
            tweets = list(cursor)
            if tweets:
                logger.info(f"📤 Retrieved {len(tweets)} tweets ready for NLP analysis.")
            return tweets
        except Exception as e:
            logger.error(f"❌ Error retrieving tweets for analysis: {e}")
            return []

    def save_bot_analysis_batch(self, analysis_results: List[Dict[str, Any]]) -> bool:
        """
        Saves a batch of bot analysis results and marks the corresponding tweets as analyzed.
        Uses a two-step bulk operation for high efficiency.
        """
        if not analysis_results:
            return False

        tweet_ids_to_update = [res['tweet_id'] for res in analysis_results]

        try:
            # Step 1: Bulk insert the new analysis results into their dedicated collection.
            # Using InsertOne is more explicit for new documents.
            analysis_ops = [InsertOne(doc) for doc in analysis_results]
            self.bot_analysis_collection.bulk_write(analysis_ops, ordered=False)

            # Step 2: Bulk update the original tweets to mark them as analyzed.
            self.tweets_collection.update_many(
                {"tweet_id": {"$in": tweet_ids_to_update}},
                {"$set": {"bot_analyzed": True, "bot_analyzed_at": datetime.now(timezone.utc)}}
            )

            logger.info(f"✅ Saved {len(analysis_results)} analysis results and updated tweet statuses.")
            return True
        except errors.BulkWriteError as bwe:
             logger.error(f"❌ Bulk write error saving analysis batch: {bwe.details}")
             return False
        except Exception as e:
            logger.error(f"❌ Unexpected error during batch save of analysis: {e}")
            return False