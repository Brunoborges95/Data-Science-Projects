from io import BytesIO
import json
from collections import defaultdict, Counter
import pprint
import time
import datetime
import boto3
import subprocess
import pandas as pd
import scipy.sparse as sparse
ATHENA_OUTPUT_LOCATION = {
    "bucket": "dr-data-team-at-iu",
    "directory": "AthenaFromDrSagemakerQueryResults-20211011/",
}


def test():
    print("utils ok")


def optimization_run_filenames(file_name,campaignid,optimization_run_timestamp):
    return "sagemaker-tests-digitalreef", f"campaignid={campaignid}/optimization_runs/optimization_run={optimization_run_timestamp}/{file_name}"

def read_from_s3(reader_function, bucket, key, s3_client, **kwargs):
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    with BytesIO(obj["Body"].read()) as f:
        f.seek(0)  # rewind the file
        r = reader_function(f, **kwargs)
    return r


def pandasdf2csv(data_object,file,*args,**kwargs):
    data_object.to_csv(file,*args,**kwargs)

def pandasdf2parquet(data_object,file,*args,**kwargs):
    data_object.to_parquet(file,*args,**kwargs)

def write_json_to_file(data_object,file,*args,**kwargs):
     file.write(json.dumps(data_object).encode('utf-8')) 
        
def save_npz(data_object,file,*args,**kwargs):
    sparse.save_npz(file,data_object,*args,**kwargs)

def save_on_s3(
    data_object, saving_function, bucket, key, s3_client, *args,**kwargs
):
    with BytesIO() as file:
        saving_function(data_object,file,*args,**kwargs)
        file.seek(0)
        s3_client.upload_fileobj(
            Fileobj=file,
            Bucket=bucket,
            Key=key,
        )


class AthenaQuerier:
    def __init__(self) -> None:
        self._executed_queries = defaultdict(dict)
        self._athena_access_initialized = False
        self._ensure_athena_access()

    def _ensure_athena_access(self) -> None:
        if self._athena_access_initialized:
            pass            
        else:
            # Create credentials
            self._athena_dr_sagemaker_session = boto3.Session()
            self._athena_client = self._athena_dr_sagemaker_session.client(
                "athena", region_name="us-east-1"
            )

    def start_acadia_query_execution(self, query_statement) -> str:
        self._ensure_athena_access()

        query_response = self._athena_client.start_query_execution(
            QueryString=query_statement,
            QueryExecutionContext={"Database": "acadia", "Catalog": "AwsDataCatalog"},
            WorkGroup="primary",
            ResultConfiguration = {"OutputLocation":f"s3://{ATHENA_OUTPUT_LOCATION['bucket']}/{ATHENA_OUTPUT_LOCATION['directory']}"}
        )

        self._executed_queries[query_response["QueryExecutionId"]]

        return query_response["QueryExecutionId"]

    def _update_queries_status(self) -> dict:

        self._ensure_athena_access()

        response = self._athena_client.batch_get_query_execution(
            QueryExecutionIds=list(self._executed_queries.keys())
        )

        query_states = {}
        for query_execution in response["QueryExecutions"]:
            self._executed_queries[
                query_execution["QueryExecutionId"]
            ] = query_execution

            query_states[query_execution["QueryExecutionId"]] = query_execution[
                "Status"
            ]["State"]
        return query_states

    def wait_for_queries_completion(self, backoff_rate=2, max_wait=7200) -> dict:

        complete_states = ["SUCCEEDED", "FAILED", "CANCELLED"]
        incomplete_states = ["QUEUED", "RUNNING"]

        start_time = datetime.datetime.now()
        queries_completed = False
        backoff = 10
        wait_num = 1

        print(
            (
                "Starting to wait for queries completion...\n"
                "Will wait at most {} seconds...\n".format(max_wait)
            )
        )
        while (
            (datetime.datetime.now() - start_time).total_seconds() < max_wait
        ) and not queries_completed:
            print(
                "Starting wait number {} at {} for {} seconds".format(
                    wait_num, datetime.datetime.now(), backoff
                )
            )
            time.sleep(backoff)
            print("Finished waiting number {}".format(wait_num))
            query_states = self._update_queries_status()
            print("Query States:")
            pprint.pprint(query_states)

            states_counter = Counter(list(query_states.values()))

            if sum(states_counter[s] for s in incomplete_states) == 0:
                queries_completed = True
            else:
                wait_num = wait_num + 1
                backoff = backoff * backoff_rate
        return query_states


def get_organization_name_from_campaign_id(campaign_id,s3_client,querier = None):

    athena_querier = AthenaQuerier() if querier is None else querier

    query = f"""
        SELECT camp.id as camp_id, org.id as org_id, org.name as org_name
        FROM "acadia"."campaign" as camp
        JOIN "acadia"."organization" as org
        ON camp.organizationid = org.id
        WHERE camp.id = '{campaign_id}';
    """


    query_id = athena_querier.start_acadia_query_execution(query)
    queries_dict = athena_querier.wait_for_queries_completion()

    query_state = queries_dict[query_id]

    if query_state != "SUCCEEDED":
        raise NotImplementedError(
            "Query state at {}, handling not yet implemented".format(query_state)
        )

    data = read_from_s3(
        pd.read_csv,
        ATHENA_OUTPUT_LOCATION["bucket"],
        ATHENA_OUTPUT_LOCATION["directory"] + query_id + ".csv",
        s3_client,
    )
    return data.loc[0,'org_name'].lower()

    
def clone_github_repo(cwd):
    subprocess.run(["git","pull"],cwd=cwd)
