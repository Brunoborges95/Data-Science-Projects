import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from lightfm import LightFM, data
import joblib
from multiprocessing.dummy import Pool as ThreadPool
import pynndescent
from recommender import Recommender
import utils
import boto3
from time import sleep
import multiprocessing

#number of unique users
N_USERS = 100000
#minimum number of devices on which the app needs to be installed to be considered in the training
N_DEVICES = 500
#number of latent factors for the recommendation model
N_LATENT_FACTORS=30
#number of recommendation model training epochs
N_EPOCHS = 10
#number of threads for parallel processing
N_THREADS = multiprocessing.cpu_count() # or os.cpu_count()
#organizations selected
ORGS = ['entel', 'claroco', 'tim brasil', 'vivo', 'telefonica peru', 'claro br', 'claro ecuador']
BUCKET = 'dr-campaign-optimization'

pool = ThreadPool(N_THREADS)
boto3_session = boto3.Session()
s3_client = boto3_session.client('s3')

def get_data_from_athena(query,athena_querier=None,retries=1):
    """
    TODO: divide this method in 2: sending query to querier and then reading results
       queries athena for the data needed. Will create the raw_data and column_index attributes
        Inputs: depends on the 
        Outputs:          
    """
    if athena_querier is None:
        athena_querier = utils.AthenaQuerier()

    query_state = None
    while (query_state != "SUCCEEDED" and retries>0):
        query_id = athena_querier.start_acadia_query_execution(query)
        queries_dict = athena_querier.wait_for_queries_completion(backoff_rate=1.2)    
        query_state = queries_dict[query_id]
        retries-=1
        if query_state != "SUCCEEDED":
            sleep(10)

    if query_state != "SUCCEEDED":
            raise NotImplementedError(
                "Max retries reached and query state at {}, handling not yet implemented".format(query_state)
            )
    else:
        raw_data = utils.read_from_s3(
        pd.read_csv,
        utils.ATHENA_OUTPUT_LOCATION["bucket"],
        utils.ATHENA_OUTPUT_LOCATION["directory"] + query_id + ".csv",
        s3_client,
    )
        return raw_data
    
def load_training_data(orgs, max_len, is_local=False):
    if isinstance(orgs, str):
        orgs=f'({orgs})'
    elif isinstance(orgs, list):
        orgs=tuple(orgs)
        
    if is_local:
      df = pd.read_csv('df_installedapps.csv')[['clientid', 'appname', 'devicemodel', 'org']].drop_duplicates()
      return df
    else:   
        query = f"""
        with devices_recent as (
            select distinct 
                clientid, org, carrier, devicemake, devicemodel, lastactivedate
            from 
                deviceinfo
            where 
                org in {orgs} and clientid in (select distinct(clientid)
                                            from "acadia".installedapps)
            order by lastactivedate desc
            limit {max_len}
        ) 
        select 
            ia.clientid, appname, di.org, ia.org, devicemodel 
        from 
            "acadia"."installedapps" ia join devices_recent di on (ia.clientid = di.clientid)
        """   
        athena_querier = utils.AthenaQuerier()
        q = get_data_from_athena(query, athena_querier=athena_querier)
        df = q[['clientid', 'appname', 'devicemodel', 'org']].drop_duplicates()
    return q

for org in ORGS:
    print(f'---------------------------LOAD TRAINING DATA-{org}----------------------------')       
    df = load_training_data(org, N_USERS, is_local=False)
    n_users_app = df.appname.value_counts().reset_index()
    n_users_app['%total'] = n_users_app.appname/df.clientid.nunique()
    n_users_app.columns = ['app', 'n_users', '%total']
    
    q2 = sum(n_users_app.n_users>=N_DEVICES)
    print(f'O número de apps instalado em mais de {N_DEVICES} devices é de {q2}.')
    relevant_apps = list(n_users_app[n_users_app.n_users>=N_DEVICES].app)
    df = df[df.appname.isin(relevant_apps)]
    
    print(f'---------------------Building sparse matrix-{org}-------------------------------')
    matrix = data.Dataset()
    matrix.fit(users=list(df.clientid.unique()), items=list(df.appname.unique()), 
                         user_features=list(df.devicemodel.unique()))
    interactions_unmasked = matrix.build_interactions(data=[[row[1], row[2]] for row in df[['clientid', 'appname']].itertuples()])
    user_features = matrix.build_user_features([[row[1], [row[2]]] for row in df[['clientid', 'devicemodel']].drop_duplicates().itertuples()], normalize=False)
    mapping = matrix.mapping()
    
    
    print(f'------------------------------TRAINING-{org}----------------------------')
    model_lightfm_warp = LightFM(no_components=N_LATENT_FACTORS, learning_rate=0.05, loss='warp')
    model_lightfm_warp.fit(interactions_unmasked[0], epochs=N_EPOCHS, verbose=True, num_threads=N_THREADS)
    nbrs_training = pynndescent.NNDescent(interactions_unmasked[0].tocsr(), metric="cosine", n_jobs=N_THREADS, verbose=True)
    
    print(f'--------------------CREATING USER SCORE DATABASE-{org}-------------------------')
    rec = Recommender(mapping, interactions_unmasked[0], relevant_apps)
    predictions_pd = rec.predict(model_lightfm_warp) 
    
    print(f'------------------------SAVE FILES-{org}---------------------------------------')
    utils.save_on_s3(predictions_pd,
               utils.pandasdf2csv,
               BUCKET,
               f'apprecommender/data_{org}/predictions_pd.csv',
               s3_client)
    
    utils.save_on_s3(mapping,
               joblib.dump,
               BUCKET,
               f'apprecommender/data_{org}/mapping.joblib',
               s3_client)
    
    utils.save_on_s3(model_lightfm_warp,
               joblib.dump,
               BUCKET,
               f'apprecommender/data_{org}/model_lightfm_warp.joblib',
               s3_client)
    
    utils.save_on_s3(nbrs_training,
               joblib.dump,
               BUCKET,
               f'apprecommender/data_{org}/nbrs_training.joblib',
               s3_client)
    
    utils.save_on_s3(relevant_apps,
               joblib.dump,
               BUCKET,
               f'apprecommender/data_{org}/relevant_apps.joblib',
               s3_client)
    
    utils.save_on_s3(interactions_unmasked[0],
                     utils.save_npz,
                     bucket=BUCKET,
                     key=f'apprecommender/data_{org}/interactions_unmasked.npz',
                     s3_client=s3_client)

