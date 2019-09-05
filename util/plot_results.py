import comet_ml
import numpy as np
import matplotlib.pyplot as plt

comet_api = comet_ml.API(rest_api_key='jHxSNRKAIOSSBP4TyRvGHfanF')

results = {'HalfCheetah-v2': {'SAC (Uniform Prior)':               [],
                              'SAC (TanhGaussian Prior)':          ['fb6f2ebadf614b1ca970bbde0fe0d04c',
                                                                    'd0e8ecf209d34aed9c480f8bac032eae'],
                              'SAC + Iterative Inference':         ['a7ffacd3c9504c70b555306d8aeb8aed'],
                              'SAC + Iterative Inference + Learned Prior': ['086ab6b9f3ff4e7b9059e144ec2fd4f7']
                              }
            }


def get_experiment_metrics_raw(experiment_key, metric):
    metric_list = []
    raw_metrics = comet_api.get_experiment_metrics_raw(experiment_key)
    for raw_metric in raw_metrics:
        if raw_metric['metricName'] == metric:
            metric_list.append(float(raw_metric['metricValue']))
    return np.array(metric_list)

def plot_results(env_name, result_dict):
    plt.figure()
    for exp_name, exp_list in result_dict.items():
        if len(exp_list) > 0:
            met_list = []
            for experiment_key in exp_list:
                cum_reward = get_experiment_metrics_raw(experiment_key, 'cumulative_reward')
                met_list.append(cum_reward)
            min_length = min([met.shape[0] for met in met_list])
            clipped_mets = np.stack([met[:min_length] for met in met_list])
            avg_cum_reward = clipped_mets.mean(0)
            plt.plot(avg_cum_reward, label=exp_name)
    plt.legend()
    plt.title(env_name)
    plt.show()


if __name__ == '__main__':
    for env_name, result_dict in results.items():
        plot_results(env_name, result_dict)
