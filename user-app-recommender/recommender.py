import tqdm
import numpy as np
import pandas as pd
import math
import scipy.sparse as sparse
from multiprocessing.dummy import Pool as ThreadPool
from itertools import repeat
import random
pool = ThreadPool(12)
from typing import List
import pynndescent
from lightfm import LightFM
import multiprocessing
N_THREADS = multiprocessing.cpu_count()

class Recommender():
    
    def __init__(self, mapping, interactions, most_popular_apps,
                 interactions_masked=None):
        """
        Parameters
        ----------
        mapping : TYPE
            It makes the conversion between customer id and customer index in sparse array.
        interactions : sparse
            Sparse array with installed apps for each client.
        most_popular_apps : list
            Ordered list of the most popular apps.
        interactions_masked : sparse, optional
            If not None, it is used for testing.
            Contains the masked applications for evaluation of
            model performance. The default is None.

        Returns
        -------
        None.

        """
        self.mapping = mapping
        self.interactions = interactions
        self.interactions_masked = interactions_masked
        self.most_popular_apps = most_popular_apps
        self.apps_all_id = [self.mapping[2][app] for app in self.most_popular_apps]
    

    def _converter_id(self, name_id, keys, inverse=False):
        """
        Given a key feature, it makes the conversion between the id
        and the indica in the sparse matrix.

        Parameters
        ----------
        name_id : string      
            Key feature. It might be 'clientid', 'devicemodel' or 'appname'.
        keys : string or list of strings       
            Key feature to be converted.
        inverse : boolean, optional
            True case converts the index to the id. The default is False.

        Raises
        ------
        Exception
            If the feature key is not one of the three selected.

        Returns
        -------
        string or list of strings.           
            converted key feature.
        """
        #name_id = {clientid, appname, devicemodel}
        if isinstance(keys, str):
            keys = [keys]
        valid_name_id = ['clientid', 'devicemodel', 'appname']
        keys_id = [] 
        for key in keys:
            if name_id not in valid_name_id:
                raise Exception(f"Choose one valid name_id in the list: {valid_name_id}")
            if inverse==False:
                if name_id=='clientid':
                    map_0 = self.mapping[0]
                    keys_id.append(map_0[key])
                if name_id=='devicemodel':
                    map_1 = self.mapping[1]
                    keys_id.append(map_1[key])
                if name_id=='appname':
                    map_2 = self.mapping[2]
                    keys_id.append(map_2[key])
            if inverse==True:
                if name_id=='clientid':
                    map_0 = {v:k for k, v in self.mapping[0].items()}
                    keys_id.append(map_0[key])
                if name_id=='devicemodel':
                    map_1 = {v:k for k, v in self.mapping[1].items()}
                    keys_id.append(map_1[key])
                if name_id=='appname':
                    map_2 = {v:k for k, v in self.mapping[2].items()}
                    keys_id.append(map_2[key]) 
        if isinstance(keys, str):
            return keys_id[0]
        else:
            return keys_id
    
    def predict(self, model, clientids=None, user_ids=None, filter_user_apps=True, 
                ranking_mode=False, k=10, return_app_id=False):
        """
        It recommends applications to selected users.
        Parameters
        ----------
        model : Trained LightFM artifact. 
            Trained artifact from a recommendation library (LightFM).
        clientids : List of strings, optional
            list of client ids for recommendation.
            The default is None. If both clientid and user are None, the function will predict for all trained users.
        user_ids : Listo of integers, optional
            List of user indices for recommendation.
            The default is None.
        filter_user_apps : boolean, optional
            Whether in the recommendation you should filter the apps that users already have installed.
            The default is True.
        ranking_mode : boolean, optional
            See the description of the function return.
            The default is False.
        k : integer, optional
            If ranking_mode is True, choose k, which provides the top k recommendations from users.
            The default is 10.
        return_app_id : boolean, optional
            Defines how apps should be displayed in the output. True case returns the id.
            If false, returns the name of the app. The default is False.

        Raises
        ------
        Exception
            If the model input is not a valid model for recommendation.

        Returns
        -------
        If Ranking_mode is False: DataFrame. If is True: dictionary
            If Ranking_mode is False, the function returns a dataframe 
                with the score of all applications for each user.
            If True, returns a dictionary containing the top k recommendations for each user.

        """
        if clientids is None and user_ids is None:
            user_rows = self.interactions.tocsr()
            user_ids = list(self.mapping[0].values())
            clientids = list(self.mapping[0].keys())
        elif clientids is None and user_ids is not None:
            user_ids = list(user_ids)
            user_rows = self.interactions.tocsr()[user_ids,:]
        else:
            user_ids = self._converter_id('clientid', clientids)
            user_rows = self.interactions.tocsr()[user_ids,:]
        if isinstance(model, LightFM):
            predictions = np.array([model.predict(int(user_id), list(set(self.apps_all_id)),
                                    num_threads=N_THREADS) for user_id in tqdm.tqdm(user_ids)])
            if filter_user_apps:
                predictions = np.where(user_rows.toarray(), np.nan, predictions)
        else:
            raise Exception("Please, provide a valid model.")
        if clientids is None and user_ids is not None:
            index = user_ids
        else:
            index = clientids
        if return_app_id:
            predictions_pd = pd.DataFrame(predictions, index=index, columns=set(self.apps_all_id))
        else:
            predictions_pd = pd.DataFrame(predictions, index=index, 
                             columns=self._converter_id('appname', set(self.apps_all_id), inverse=True))
        if ranking_mode:
            predictions_ranked = {}
            for i in predictions_pd.index:
                predictions_ranked[i] = predictions_pd.loc[i].sort_values(ascending=False)[:k]
            return predictions_ranked
        else:
            return predictions_pd
        
        
    def predict_baseline(self, clientids=None, user_ids=None, users_apps: List[List[str]]=None, baseline ='popularity',
                        filter_user_apps=True, ranking_mode=False, k=10, return_app_id=False):
        """
        It recommends applications to selected users using baselines.
        Parameters
        ----------
        clientids : List of strings, optional
            list of client ids for recommendation. If None, None is 
                necessary or provide the client index or the list of apps. The default is None.
        user_ids : List of integers, optional
            List of user indices for recommendation. If None, it is necessary or provide the client id or
            the list of apps. The default is None.
        users_apps : List of Lists, optional
            List of apps for each user. If None, either client ids or user indices must be provided.
            The default is None.
        baseline : 'popularity' or 'random', optional
            Selection of the Baseline. The default is 'popularity'.
        filter_user_apps : boolean, optional
            Whether the existed user apps should be filtered or not. The default is True.
        ranking_mode : boolean, optional
            See the description of the function return.
            The default is False.
        k : integer, optional
            If ranking_mode is True, choose k, which provides the top k recommendations from users.
            The default is 10.
        return_app_id : boolean, optional
            Defines how apps should be displayed in the output. True case returns the id.
            If false, returns the name of the app. The default is False.

        Raises
        ------
        Exception
            If the baseline is different from 'popularity' or 'random;
            If the user_apps, user_ids or client ids parameters are all None.
        Returns
        -------
        If Ranking_mode is False: DataFrame. If is True: dictionary
            If Ranking_mode is False, the function returns a dataframe 
                with the score of all applications for each user.
            If True, returns a dictionary containing the top k recommendations for each user.
        """
        #clientid --> user_id
        valid_baselines = ['popularity', 'random']
        if baseline not in valid_baselines:
            raise Exception(f"Choose one valid baseline in the list: {valid_baselines}")
        #by user_apps
        if users_apps is not None:
            apps_user_ids = [self._converter_id('appname', user_apps) for user_apps in users_apps]
            num_users = len(apps_user_ids)
            index=None
        else:
            #by clientids
            if user_ids is None and clientids is not None:
                self.user_ids = self._converter_id('clientid', clientids)
                user_ids =  self.user_ids
                index = clientids
            #by userids
            elif user_ids is not None and clientids is None:
                self.user_ids = user_ids
                index = user_ids
            num_users = len(user_ids)
            user_rows = self.interactions.tocsr()[user_ids,:]
            apps_user_ids = [list(np.where(user_row.getnnz(axis=0) > 0)[0]) for user_row in user_rows]
            if user_ids is None and clientids is None and users_apps is None:
                raise Exception('Provide or clientid or user_id or user_apps.')    
        if baseline=='random':
            s = np.array([np.random.random() for k in range(len(self.apps_all_id))])
        if baseline=='popularity':
            s = np.array([1/math.log(k+2) for k in range(len(self.apps_all_id))])
        scores = np.matlib.repmat(s,num_users,1)
        if return_app_id:
            apps = self.apps_all_id
        else:
            apps = self.most_popular_apps
        if filter_user_apps:
            scores_masked = np.where([np.isin(self.apps_all_id, i) for i in apps_user_ids], np.nan, scores)                    
            predictions_pd = pd.DataFrame(scores_masked, index=index, 
                              columns=apps)
        else: 
            predictions_pd = pd.DataFrame(scores, index=index, 
                              columns=apps)
        if ranking_mode:
            predictions_ranked = {}
            for i in predictions_pd.index:
                predictions_ranked[i] = predictions_pd.loc[i].sort_values(ascending=False)[:k]
            return predictions_ranked
        else:
            return predictions_pd
        
    def predict_similarity_based(self, nbrs_model, model, users_apps: List[List[str]], n_neighbors=5,
                                ranking_mode=False, k=10):
        """
        It recommends applications to selected users using similarity with users of training data.
        Parameters
        ----------
        nbrs_model : pynndescent.pynndescent_.NNDescent artifact      
            Approximate Nearest-neighbors search training artifact.
        model : Trained LightFM artifact. 
            Trained artifact from a recommendation library (LightFM).
        users_apps : List of Lists, optional
            List of apps for each user. If None, either client ids or user indices must be provided.
            The default is None.
        n_neighbors : Int, optional
            Number of neighbors. The default is 5.
         ranking_mode : boolean, optional
            See the description of the function return.
            The default is False.
        k : integer, optional
            If ranking_mode is True, choose k, which provides the top k recommendations from users.
            The default is 10.
        Raises
        ------
        Exception
            If the approximate nearest neighbor search model is not a valid model for recommendation.
            If the model input is not a valid model for recommendation.

        Returns
        -------
        If Ranking_mode is False: DataFrame. If is True: dictionary
            If Ranking_mode is False, the function returns a dataframe 
                with the score of all applications for each user.
            If True, returns a dictionary containing the top k recommendations for each user.
        """
        apps_id = np.array([self._converter_id('appname', user_apps) for user_apps in users_apps])
        J = np.hstack(apps_id) #position in column
        V = np.ones(len(J)).astype(int) #fll with 1
        I = np.hstack([np.repeat(i,len(v)) for i,v in enumerate(apps_id)]) #position in row
        sparse_users = sparse.csr_matrix((V,(I,J)))
        if isinstance(nbrs_model, pynndescent.pynndescent_.NNDescent):
            users_nbr, users_nbr_distances = nbrs_model.query(sparse_users, k=n_neighbors)
        else:
            raise Exception("Provide a valid pynndescent artifact")
        #dict_rec = {r:0 for r in set(self.most_popular_apps)-set(user_apps)}
        if isinstance(model, LightFM):
            df = self.predict(model, user_ids=list(users_nbr.flatten()), filter_user_apps=False)
        else:
            raise Exception("Please, provide a valid model.")
        df = df.assign(user =  np.repeat(range(0,len(users_nbr)),n_neighbors))\
                  .groupby('user').mean()
        temp = np.where([np.isin(df.columns, i) for i in users_apps], np.nan, df)
        predictions_pd = pd.DataFrame(temp, columns=df.columns)
        if ranking_mode:
            predictions_ranked = {}
            for i in predictions_pd.index:
                predictions_ranked[i] = predictions_pd.loc[i].sort_values(ascending=False)[:k]
            return predictions_ranked
        else:
            return predictions_pd
    
  
    def _score(self, predicted, actual, metric):
        """
        Parameters
        ----------
        predicted : List
            List of predicted apps.
        actual : List
            List of masked apps.
        metric : 'precision' or 'ndcg'
            A valid metric for recommendation.
        Raises
        -----
        Returns
        -------
        m : float
            score.
        """
        valid_metrics = ['precision', 'ndcg']
        if metric not in valid_metrics:
            raise Exception(f"Choose one valid baseline in the list: {valid_metrics}")
        if metric == 'precision':
            m = np.mean([float(len(set(predicted[:k]) 
                                               & set(actual))) / float(k) 
                                     for k in range(1,len(actual)+1)])
        if metric == 'ndcg':
            v = [1 if i in actual else 0 for i in predicted]
            v_2 = [1 for i in actual]
            dcg = sum([(2**i-1)/math.log(k+2,2) for (k,i) in enumerate(v)])
            idcg = sum([(2**i-1)/math.log(k+2,2) for (k,i) in enumerate(v_2)])
            m = dcg/idcg
        return m
            
    def average_precision(self):
        """
        Calculate the mean average precision.
        Avaliable only in test mode.
        Returns
        -------
        avg_precision : float
        """
        if self.interactions_masked is None:
            print("Please, for evaluate the precision, its necessary to provide the masked interaction.")
        if self.interactions_masked is not None:
            user_row = self.interactions_masked.tocsr()[self.user_id,:]
            apps_masked_user_id = list(np.where(user_row.getnnz(axis=0) > 0)[0])
            avg_precision = self._score(self.recommendation_id, apps_masked_user_id, 'precision')
            return avg_precision

    def ndcg(self):
        """
        Calculate the NDCG. 
        Avaliable only in test mode.
        Returns
        -------
        ndcg : float
        """
        if self.interactions_masked is None:
            print("Please, for evaluate the NDCG, its necessary to provide the masked interaction.")
        if self.interactions_masked is not None:
            user_row = self.interactions_masked.tocsr()[self.user_id,:]
            apps_masked_user_id = list(np.where(user_row.getnnz(axis=0) > 0)[0])
            ndcg = self._score(self.recommendation_id, apps_masked_user_id, 'nfcg')
            return ndcg
    
    def evaluating_precision_model_by_k(self, models, models_name, k_max,
                                        n_sample=100, baselines=['popularity']):
        '''
        Calculate the precision@k metric, that is,
        How many relevant items are present in the top-k recommendations of your system.
        Avaliable only in test mode.
        Parameters
        ----------
        models : List
            List of artifacts models for recommendation.
        models_name : List of strings
            List of the model's name.
        k_max : int
            Maximum k then be estimated.
        n_sample : int, optional
            Number of users to estimate the average precision.The default is 100.
        baselines : List, optional
            Baselines for evaluation. If None, the baseline precision is not evaluated.
            The default is ['popularity'].
        Returns
        -------
        DataFrame
            For each k, return the average precision for each model selected.
        '''
        sample_users_id = random.sample(set(self.mapping[0].values()),n_sample)
        evaluations = {}
        k_max+=1
        k_range = range(2,k_max)
        if baselines is not None:
            for b in baselines:
                models_name.append(b)
                models.append(b)
        for name in models_name:
            evaluations[f'{name}'] = {}
            for k in k_range:
                evaluations[f'{name}'][f'{k}'] = []
            
        user_rows = self.interactions_masked.tocsr()[sample_users_id,:]
        apps_masked_user_ids = [list(np.where(user_row.getnnz(axis=0) > 0)[0]) for user_row in user_rows]
        for name, model in zip(models_name, models):
            if isinstance(model, str):
                rec = self.predict_baseline(user_ids=sample_users_id, 
                                    ranking_mode=True, baseline=name, k=k_max, return_app_id=True)
            if isinstance(model, LightFM):
                rec = self.predict(model, user_ids=sample_users_id, 
                                    k=k_max, ranking_mode=True, return_app_id=True)
            def _f_1(user_id, apps, pbar):
                pbar.update(1)
                k_user = len(apps)
                for k in range(2, min(k_user, k_max)): 
                    correct_apps_id = set(rec[user_id].index[:k]) & set(apps)
                    precision = float(len(correct_apps_id)) / float(k)
                    evaluations[name][f'{k}'].append(precision)
        
            with tqdm.tqdm(total=len(sample_users_id)) as pbar:
                 pool.starmap(_f_1, zip(sample_users_id, apps_masked_user_ids, repeat(pbar)))       
        precision_evaluation_avg = {}
        for name in models_name:
            precision_evaluation_avg[name] = [np.mean(evaluations[name][f'{k}']) for k in k_range]
        return pd.DataFrame(precision_evaluation_avg, index=k_range) 
    
    def evaluation_model(self, models, models_name, n_sample=100, use_baseline=True):
        """
        calculate de the MAP - Mean average precision and the mean NDCG for a sample of users.
        Avaliable only in test mode.
        Parameters
        ----------
         models : List
            List of artifacts models for recommendation.
        models_name : List of strings
            List of the model's name.
        k_max : int
            Maximum k then be estimated.
        n_sample : int, optional
            Number of users to estimate the average precision.The default is 100.
        use_baseline : boolean, optional
            If use the populary or not. The default is True.
        Raises
        ------
        Exception
            If the model input is not a valid model for recommendation..
        Returns
        -------
        dict
            The average mean precision and the average NDCG for each model.
        """
        sample_users_id = random.sample(set(self.mapping[0].values()),n_sample)
        avg_precision = {}
        avg_ndcg = {}
        if use_baseline:
            models_name.append('popularity')
            models.append('popularity')
        for name, model in zip(models_name, models):
            precision_user = []
            ndcg_user = []
            user_rows = self.interactions_masked.tocsr()[sample_users_id,:]
            apps_masked_user_ids = [list(np.where(user_row.getnnz(axis=0) > 0)[0]) for user_row in user_rows]
            if isinstance(model, LightFM):
                predictions_pd = self.predict(model, user_ids=sample_users_id, 
                                              ranking_mode=True, k=100, return_app_id=True)
            elif isinstance(model, str):
                predictions_pd = self.predict_baseline(user_ids=sample_users_id, baseline ='popularity',
                        ranking_mode=True, k=100, return_app_id=True)
            else:
                raise Exception("Please, provide a valid model.")
            def _f_2(i, user_id, pbar):
                pbar.update(1)
                predicted = predictions_pd[user_id].index
                actual = apps_masked_user_ids[i]
                if len(actual)>2:
                    precision_user.append(self._score(predicted, actual, 'precision'))
                    ndcg_user.append(self._score(predicted, actual, 'ndcg'))
    
            with tqdm.tqdm(total=len(sample_users_id)) as pbar:
                pool.starmap(_f_2, zip(range(len(sample_users_id)), sample_users_id, repeat(pbar)))
            avg_precision[name] = np.mean(precision_user)
            avg_ndcg[name] = np.mean(ndcg_user)
        return {'avg_precision':avg_precision, 'avg_ndcg':avg_ndcg}