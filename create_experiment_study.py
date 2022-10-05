import mlflow
import optuna
from optuna.samplers import TPESampler

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='name for study in optuna and experiment in mlflow')
    parser.add_argument('--tracking_uri', default='http://mlflow.172.26.62.216.nip.io', help='uri where mlflow_server is running')
#    parser.add_argument('--tracking_uri', default='http://172.26.62.216:9000', help='uri where mlflow_server is running')
#    parser.add_argument('--postgres_db', default='postgresql://mlflow_user:mlflow@172.26.62.216:5432/optuna' , help='connection string to where optuna stores studies')
    parser.add_argument('--postgres_db', default='postgresql://mlflow_user:mlflow@172.26.62.216:6543/optuna' , help='connection string to where optuna stores studies')
    
    args = parser.parse_args()
    
    mlflow.set_tracking_uri(args.tracking_uri)

    storage=args.postgres_db
    mlflow.set_experiment(args.name)
    study = optuna.create_study(study_name=args.name,
                                direction="maximize",
                                storage=storage,
                                sampler=TPESampler(n_startup_trials=8),
                                load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=20,
                                                                   n_warmup_steps=15,
                                                                   interval_steps=10,
                                                                   n_min_trials=10))