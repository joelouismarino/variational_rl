import matplotlib.pyplot as plt
import scipy.stats as ss
import numpy as np
import pickle
import os


def plot_dashboard(results, dir, plot=True):
    """
    Plots a dashboard of the agent's performance as a series of images.

    Args:
        result (dict): contains episode performance metrics
        dir (str): path to where dashboard plots should be saved
    """
    if plot:
        n_steps = len(results['observation'])
        # plot each of the dashboard frames
        for step in range(n_steps):
            fig = plt.figure(figsize=(17, 10))
            img_obs = len(results['observation'][step].size()) == 3
            state_dim = len(results['observation'][step])
            # plot the observation
            plt.subplot(4, 3, 1)
            if img_obs:
                observation = results['observation'][step].numpy().transpose(1, 2, 0).clip(0, 1)
                plt.imshow(observation)
                plt.axis('off')
            else:
                # plot mujoco
                observation = np.reshape(results['observation'][step].numpy(), (1, state_dim))
                im = plt.imshow(observation, cmap = 'Greys', vmin = -2, vmax = 2)
                plt.tick_params(left = 'off', bottom = 'off', labelleft='off', labelbottom='off')
                plt.colorbar(im, orientation='horizontal')

            plt.title('Observation', fontsize=10)

            # plot the prediction
            plt.subplot(8, 6, 3)
            if 'observation' in results['distributions'].keys():
                if img_obs:
                    prediction = results['distributions']['observation']['pred']['loc'][step].numpy().transpose(1, 2, 0)
                    plt.imshow(prediction)
                    plt.axis('off')
                else:
                    # plot mujoco
                    prediction = np.reshape(results['distributions']['observation']['pred']['loc'][step].numpy(), (1, state_dim))
                    plt.imshow(prediction, cmap='Greys', vmin = -2, vmax = 2)
                    plt.tick_params(left='off', bottom='off', labelleft='off', labelbottom='off')
                plt.title('Prediction', fontsize=10)

            # plot the prediction error
            plt.subplot(8, 6, 4)
            if 'observation' in results['distributions'].keys():
                if img_obs:
                    prediction_error = (observation - prediction + 1.) / 2
                    plt.imshow(prediction_error)
                    plt.axis('off')
                else:
                    prediction_error = (observation - prediction)
                    plt.imshow(prediction_error, cmap='Greys', vmin = -2, vmax = 2)
                    plt.tick_params(left='off', bottom='off', labelleft='off', labelbottom='off')
                plt.title('Prediction Error', fontsize=10)

            # plot the reconstruction
            plt.subplot(8, 6, 9)
            if 'observation' in results['distributions'].keys():
                if img_obs:
                    reconstruction = results['distributions']['observation']['recon']['loc'][step].numpy().transpose(1, 2, 0)
                    plt.imshow(reconstruction)
                    plt.axis('off')
                else:
                    reconstruction = np.reshape(results['distributions']['observation']['recon']['loc'][step].numpy(), (1, state_dim))
                    plt.imshow(reconstruction, cmap='Greys', vmin = -2, vmax = 2)
                    plt.tick_params(left='off', bottom='off', labelleft='off', labelbottom='off')
                plt.title('Reconstruction', fontsize=10)

            # plot the reconstruction error
            plt.subplot(8, 6, 10)
            if 'observation' in results['distributions'].keys():
                if img_obs:
                    reconstruction_error = (observation - reconstruction + 1.) / 2
                    plt.imshow(reconstruction_error)
                    plt.axis('off')
                else:
                    reconstruction_error = (observation - reconstruction)
                    plt.imshow(reconstruction_error, cmap='Greys', vmin = -2, vmax = 2)
                    plt.tick_params(left='off', bottom='off', labelleft='off', labelbottom='off')
                plt.title('Reconstruction Error', fontsize=10)

            # plot the reward conditional likelihood distribution
            plt.subplot(8, 3, 3)
            if 'reward' in results['distributions'].keys():
                reward = results['reward'][step].item()
                reward_cll_pred_loc = results['distributions']['reward']['pred']['loc'][step].item()
                reward_cll_pred_scale = results['distributions']['reward']['pred']['scale'][step].item()
                reward_cll_recon_loc = results['distributions']['reward']['recon']['loc'][step].item()
                reward_cll_recon_scale = results['distributions']['reward']['recon']['scale'][step].item()
                x = np.linspace(-1, 2, 1000)
                reward_pred = ss.norm.pdf(x, reward_cll_pred_loc, reward_cll_pred_scale)
                reward_recon = ss.norm.pdf(x, reward_cll_recon_loc, reward_cll_recon_scale)
                plt.plot(x, reward_pred, label='Prediction')
                plt.plot(x, reward_recon, label='Reconstruction')
                point = plt.plot([reward], [0], 'gD', label='Reward')[0]
                point.set_clip_on(False)
                plt.legend(loc='upper center', ncol=3)
                plt.title('Reward Prediction and Reconstruction')
                plt.xlim(-1, 2)
                plt.ylim(0, 5)

            # plot the done conditional likelihood distribution
            plt.subplot(8, 3, 6)
            if 'done' in results['distributions'].keys():
                done = results['done'][step].item()
                done_cll_pred_prob = results['distributions']['done']['pred']['probs'][step].numpy()
                done_cll_recon_prob = results['distributions']['done']['recon']['probs'][step].numpy()
                if done_cll_pred_prob.shape[0] == 2:
                    # categorical
                    done_cll_pred_prob = done_cll_pred_prob[1]
                    done_cll_recon_prob = done_cll_recon_prob[1]
                else:
                    # bernoulli
                    done_cll_pred_prob = done_cll_pred_prob[0]
                    done_cll_recon_prob = done_cll_recon_prob[0]
                plt.bar([-0.125, 0.875], [1. - done_cll_pred_prob, done_cll_pred_prob], 0.25, label='Prediction')
                plt.bar([0.125, 1.125], [1. - done_cll_recon_prob, done_cll_recon_prob], 0.25, label='Reconstruction')
                point = plt.plot([done], [0], 'gD', label='Done')[0]
                point.set_clip_on(False)
                plt.legend(loc='upper center', ncol=3)
                plt.title('Done Prediction and Reconstruction')
                plt.ylim(0, 2)
                plt.xticks([0,1])

            # plot the action distribution
            discrete_actions = 'probs' in results['distributions']['action']['prior']
            plt.subplot(8, 3, 7)
            if discrete_actions:
                action = np.argmax(results['action'][step].numpy())
                action_prior_probs = results['distributions']['action']['prior']['probs'][step].numpy()
                action_approx_post_probs = results['distributions']['action']['approx_post']['probs'][step].numpy()
                plt.bar(np.arange(action_prior_probs.shape[0]) - 0.125 , action_prior_probs, 0.25, label='Prior')
                plt.bar(np.arange(action_approx_post_probs.shape[0]) + 0.125 , action_approx_post_probs, 0.25, label='Approx. Post.')
                point = plt.plot([action], [0], 'gD', label='Action')[0]
                point.set_clip_on(False)
                plt.ylim(0, 2)
                plt.xticks(np.arange(len(action_approx_post_probs)))
                plt.legend(loc='upper center', ncol=3)
            else:
                pass
            plt.title('Action')

            # plot state KL-divergence over time
            plt.subplot(8, 3, 8)
            state_kl = results['metrics']['state']['kl'][:step+1].numpy()
            plt.plot(state_kl)
            plt.ylim(0, results['metrics']['state']['kl'].max().item() + 0.5)
            plt.xlim(0, n_steps)
            plt.title('State KL')

            # plot the state inference improvement over time
            plt.subplot(8, 3, 9)
            if len(results['inf_imp']['state']) > 0:
                state_inf_imp = results['inf_imp']['state'][:step+1].numpy()
                plt.plot(state_inf_imp)
                plt.ylim(results['inf_imp']['state'].min().item() - 0.5,
                         results['inf_imp']['state'].max().item() + 0.5)
                plt.xlim(0, n_steps)
                plt.title('State Inf. Imp.')

            # plot the reward over time
            plt.subplot(8, 3, 10)
            reward = results['reward'][:step+1].numpy()
            plt.plot(reward)
            plt.ylim(results['reward'].min().item() - 0.5, results['reward'].max().item() + 0.5)
            plt.xlim(0, n_steps)
            plt.title('Reward')

            # plot the action KL-divergence over time
            plt.subplot(8, 3, 11)
            action_kl = results['metrics']['action']['kl'][:step+1].numpy()
            plt.plot(action_kl)
            plt.ylim(0, results['metrics']['action']['kl'].max().item() + 0.5)
            plt.xlim(0, n_steps)
            plt.title('Action KL')

            # plot the action inference improvement over time
            plt.subplot(8, 3, 12)
            if len(results['inf_imp']['action']) > 0:
                action_inf_imp = results['inf_imp']['action'][:step+1]
                plt.plot(action_inf_imp)
                plt.ylim(results['inf_imp']['action'].min().item() - 0.5,
                         results['inf_imp']['action'].max().item() + 0.5)
                plt.xlim(0, n_steps)
                plt.title('Action Inf. Imp.')

            # plot the value estimate over time
            plt.subplot(8, 3, 13)
            values = results['value'][:step+1].numpy()
            plt.plot(values)
            plt.ylim(results['value'].min().item() - 0.5,
                     results['value'].max().item() + 0.5)
            plt.xlim(0, n_steps)
            plt.title('Value Estimates')

            # plot the observation info gain over time
            plt.subplot(8, 3, 14)
            if 'observation' in results['metrics'].keys():
                obs_info_gain = results['metrics']['observation']['info_gain'][:step+1].numpy()
                plt.plot(obs_info_gain)
                plt.ylim(results['metrics']['observation']['info_gain'][:-1].min().item() - 0.5,
                         results['metrics']['observation']['info_gain'].max().item() + 0.5)
                plt.xlim(0, n_steps)
                plt.title('Obs. Info. Gain')

            # plot the observation conditional log-likelihood over time
            plt.subplot(8, 3, 15)
            if 'observation' in results['metrics'].keys():
                obs_cll = results['metrics']['observation']['cll'][:step+1].numpy()
                plt.plot(obs_cll)
                plt.ylim(results['metrics']['observation']['cll'][:-1].min().item() - 0.5,
                         results['metrics']['observation']['cll'].max().item() + 0.5)
                plt.xlim(0, n_steps)
                plt.title('Obs. Cond. Log-Likelihood')

            # plot the advantage estimate over time
            plt.subplot(8, 3, 16)
            if len(results['advantage']) > 0:
                advantages = results['advantage'][:step+1].numpy()
                plt.plot(advantages)
                plt.ylim(results['advantage'].min().item() - 0.5,
                         results['advantage'].max().item() + 0.5)
                plt.xlim(0, n_steps)
            plt.title('Advantage Estimates')

            # plot the reward info gain over time
            plt.subplot(8, 3, 17)
            if 'reward' in results['metrics'].keys():
                reward_info_gain = results['metrics']['reward']['info_gain'][:step+1].numpy()
                plt.plot(reward_info_gain)
                plt.ylim(results['metrics']['reward']['info_gain'].min().item() - 0.5,
                         results['metrics']['reward']['info_gain'].max().item() + 0.5)
                plt.xlim(0, n_steps)
                plt.title('Reward Info. Gain')

            # plot the reward conditional log-likelihood over time
            plt.subplot(8, 3, 18)
            if 'reward' in results['metrics'].keys():
                reward_cll = results['metrics']['reward']['cll'][:step+1].numpy()
                plt.plot(reward_cll)
                plt.ylim(results['metrics']['reward']['cll'].min().item() - 0.5,
                         results['metrics']['reward']['cll'].max().item() + 0.5)
                plt.xlim(0, n_steps)
                plt.title('Reward Cond. Log-Likelihood')

            # plot the Monte Carlo return over time
            plt.subplot(8, 3, 19)
            if len(results['return']) > 0:
                returns = results['return'][:step+1].numpy()
                plt.plot(returns)
                plt.ylim(results['return'].min().item() - 0.5,
                         results['return'].max().item() + 0.5)
                plt.xlim(0, n_steps)
                plt.title('Discounted Return')

            # plot the done info gain over time
            plt.subplot(8, 3, 20)
            if 'done' in results['metrics'].keys():
                done_info_gain = results['metrics']['done']['info_gain'][:step+1].numpy()
                plt.plot(done_info_gain)
                plt.ylim(results['metrics']['done']['info_gain'].min().item() - 0.5,
                         results['metrics']['done']['info_gain'].max().item() + 0.5)
                plt.xlim(0, n_steps)
                plt.title('Done Info. Gain')

            # plot the done conditional log-likelihood over time
            plt.subplot(8, 3, 21)
            if 'done' in results['metrics'].keys():
                done_cll = results['metrics']['done']['cll'][:step+1].numpy()
                plt.plot(done_cll)
                plt.ylim(results['metrics']['done']['cll'].min().item() - 0.5,
                         results['metrics']['done']['cll'].max().item() + 0.5)
                plt.xlim(0, n_steps)
                plt.title('Done Cond. Log-Likelihood')

            fig.canvas.draw()
            plt.tight_layout()

            # save the figure
            fig.savefig(os.path.join(dir, 'step_' + str(step) + '.png'))

            plt.close(fig)

    # save results to file
    pickle.dump(results, open(os.path.join(dir, 'results.p'), 'wb'))
