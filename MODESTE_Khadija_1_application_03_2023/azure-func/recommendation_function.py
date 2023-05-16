from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd



def content_based_filtering(user_preferences, embeddings_matrix, user_data, article_metadata, n_recommendations = 5):

  # Définir les articles que le user a déja lu
  user_id_clicked_articles = user_data.filter(items = ['user_id', 'click_article_id']).query(f'user_id == {user_preferences}')

  # Choisir un article au hasard 
  user_choised_article_id = np.random.choice(user_id_clicked_articles['click_article_id'], size=1, replace=False)[0]

  # Calcul de la similarité cosinus entre les préférences de l'utilisateur et les embeddings des éléments
  user_embeddings_row = embeddings_matrix[user_choised_article_id, :].reshape(1,-1)
  new_embedding_matrix = np.delete(embeddings_matrix, user_choised_article_id, axis = 0)
  similarities = cosine_similarity(new_embedding_matrix, user_embeddings_row).reshape(1,-1)[0]

  # Récupération des indices des éléments les plus similaires
  sorted_similarities = np.sort(similarities)[::-1][:n_recommendations]
  top_indices = np.argsort(similarities)[::-1][:n_recommendations]
  recommandations_df = article_metadata.iloc[top_indices, :-1]

  # Affichage
  print("Recommending " + str(n_recommendations) + " products similar to " + str(user_choised_article_id) + "...")   
  print("-------")     
  for rec, similarity in zip(top_indices, sorted_similarities): 
      print("Recommended: " + str(rec) + " (cosinus similarity:" +      str(similarity) + ")")
    
  # Retourne les indices des éléments recommandés
  return top_indices