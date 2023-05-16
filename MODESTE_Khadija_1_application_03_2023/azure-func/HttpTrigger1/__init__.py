import json
import os
import pandas as pd
import logging
import azure.functions as func
from io import BytesIO
from recommendation_function import content_based_filtering


def main(req: func.HttpRequest, metadatablob: func.InputStream, mergeblob: func.InputStream) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    user_id = req.params.get('user_id')
    # Télécharger le contenu du blob en mémoire
    metadata = metadatablob.read()
    bytes_stream2 = BytesIO(metadata)
    merge = mergeblob.read()
    bytes_stream3 = BytesIO(merge)


    # Lire le contenu du blob dans un DataFrame Pandas
    articles_embeddings = pd.read_pickle("data/articles_embeddings.pickle")
    articles_metadata = pd.read_csv(bytes_stream2)        
    merge_user_article = pd.read_csv(bytes_stream3)

    if not user_id:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            user_id = req_body.get('user_id')

    if user_id:


        # Appel de la fonction de recommendation qui retourne une liste d'articles
        recommand = content_based_filtering(user_id, 
                                            embeddings_matrix = articles_embeddings,
                                            article_metadata = articles_metadata,
                                            user_data = merge_user_article).tolist()

        # Convertir la liste d'articles en json
        response_body = json.dumps({'user_id' : user_id,
                         'recommandation_articles' : recommand})
        
        # Retourne la liste d'articles au format json
        return func.HttpResponse(response_body, mimetype='application/json')
    
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )
