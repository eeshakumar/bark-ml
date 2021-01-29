from collections import deque
import numpy as np
import torch
import matplotlib.pyplot as plt


def update_params(optim, loss, networks, retain_graph=False, grad_cliping=None, count=0):
  optim.zero_grad()
  # loss.register_hook(lambda grad: print("Loss", grad))
  # loss.retain_grad()
  # plot_grad_flow(count, networks[0].named_parameters())
  # for p in networks[0].parameters():
  #   p.retain_grad = True
  loss.backward(retain_graph=retain_graph)
  # plot_grad_flow(count, networks[0].named_parameters())
  # Clip norms of gradients to stebilize training.
  if grad_cliping:
    for net in networks:
      torch.nn.utils.clip_grad_norm_(net.parameters(), grad_cliping)
  optim.step()
  return loss.retain_grad()


def plot_grad_flow(i, named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        # p.retain_grad = True
        # print("Parameter", n, p, p.grad)
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.abs().mean().detach().numpy())
            max_grads.append(p.abs().max().detach().numpy())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([plt.Line2D([0], [0], color="c", lw=4),
                plt.Line2D([0], [0], color="b", lw=4),
                plt.Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(f"{i}_grad_flow.png")


def disable_gradients(network):
  # Disable calculations of gradients.
  for param in network.parameters():
    param.requires_grad = False


def get_onehot(input_list, columns):
  if isinstance(input_list, torch.Tensor):
    input_list = input_list.squeeze().type(torch.int8).cpu().detach().numpy()
  rows = input_list.shape[0]
  input_onehot = np.zeros((rows, columns))
  input_onehot[np.arange(rows), input_list] = 1.
  return input_onehot


def get_margin_loss(actions, total_actions, is_demos, margin, device):
  margins = (torch.ones(total_actions, total_actions) - torch.eye(total_actions)) * margin
  sampled_batch_margins = margins[actions.long()] 
  return sampled_batch_margins.squeeze().to(device)


def calculate_huber_loss(td_errors, kappa=1.0):
  return torch.where(td_errors.abs() <= kappa, 0.5 * td_errors.pow(2),
                     kappa * (td_errors.abs() - 0.5 * kappa))


def calculate_supervised_margin_classification_loss(current_q, actions, predicted_actions, total_actions, is_demos, device, 
                                                    margin=0.8):
  """supervised margin loss to force Q value of all non expert actions to be lower"""
  sampled_batch_margin_loss = get_margin_loss(actions, total_actions, is_demos, margin, device)
  assert sampled_batch_margin_loss.shape == current_q.shape
  q1 = torch.max(current_q + sampled_batch_margin_loss, dim=1)[0]
  q2 = torch.diag(current_q[torch.arange(current_q.size(0)), actions.long()])
  q1 = q1.reshape(actions.shape)
  q2 = q2.reshape(actions.shape)
  assert q1.shape == q2.shape
  loss = is_demos * (q1 - q2)
  # net loss is mean of batch loss
  assert loss.shape == actions.shape
  return loss.mean()


def calculate_supervised_classification_quantile_loss(actions, states, online_net, taus, state_embeddings,
                                                      next_state_embeddings, is_demos, total_actions, device,
                                                      supervised_margin_weight=0.5, expert_margin=0.8):
  # #get q values from target network
  # curr_target_q = target_net.calculate_q(states=states)
  # print("Calculated q", curr_target_q.shape)

  # #get greedy actions from current q values
  # curr_target_q_actions = torch.argmax(curr_target_q, dim=1, keepdim=True)

  # #get quantiles for all actions
  # curr_target_quantiles = target_net.calculate_quantiles(taus, 
  #                                                           state_embeddings=state_embeddings)

  # # print("Current target quantiles", curr_target_quantiles.shape)
  # curr_target_quantiles_by_action = evaluate_quantile_at_action(
  #   curr_target_quantiles,
  #   curr_target_q_actions
  # )

  # #q value is then the mean of the quantiles Q(s, a)
  # curr_target_quantiles_by_action_q = curr_target_quantiles_by_action.mean(axis=1)

  # curr_target_quantiles_q = curr_target_quantiles.mean(axis=1)
  sampled_batch_margin_loss = get_margin_loss(actions, total_actions, is_demos, expert_margin, device)
  # print("Sampled large margin loss", sampled_batch_margin_loss)
  # q1, _ = torch.max(curr_target_quantiles_by_action_q + sampled_batch_margin_loss, axis=1)
  # print("Curr target quantiles q max", q1.shape)

  weights = supervised_margin_weight * is_demos.squeeze()
  # print("Calculated weights", weights)
  # q2 = torch.diag(curr_target_quantiles_q[torch.arange(curr_target_quantiles_q.size(0)), actions.long()])

  current_sa_quantiles = online_net.calculate_quantiles(taus, state_embeddings=state_embeddings)
  # print(current_sa_quantiles.shape)
  q = current_sa_quantiles.mean(dim=1)
  # print("Actions ", actions)
  # print("Q values", q)
  # print("Actions and q", actions, q)
  loss = calculate_expert_loss(q, sampled_batch_margin_loss, is_demos, actions, weights)
  return loss.mean()


def calculate_expert_loss(q, sampled_batch_margin_loss, is_demos, actions, weights):
  q1, _ = torch.max(q + sampled_batch_margin_loss, axis=1)
  expert_actions = is_demos * actions
  q2 = q.gather(1, expert_actions.long()).squeeze()
  loss = (q1 - q2) 
  loss = weights * loss
  return loss


def calculate_l2_reg_loss(network, device, weight=1e-5):
  l2_reg_loss = torch.tensor(0., requires_grad=True, device=device)
  for W in network.parameters():
    l2_reg_loss = l2_reg_loss + torch.pow(W, 2).sum()
  return weight * l2_reg_loss


def calculate_quantile_huber_loss(td_errors, taus, weights=None, kappa=1.0):
  assert not taus.requires_grad
  batch_size, N, N_dash = td_errors.shape

  # Calculate huber loss element-wisely.
  element_wise_huber_loss = calculate_huber_loss(td_errors, kappa)
  assert element_wise_huber_loss.shape == (batch_size, N, N_dash)

  # Calculate quantile huber loss element-wisely.
  element_wise_quantile_huber_loss = torch.abs(taus[..., None] - (
      td_errors.detach() < 0).float()) * element_wise_huber_loss / kappa
  assert element_wise_quantile_huber_loss.shape == (batch_size, N, N_dash)

  # Quantile huber loss.
  batch_quantile_huber_loss = element_wise_quantile_huber_loss.sum(dim=1).mean(
      dim=1, keepdim=True)
  assert batch_quantile_huber_loss.shape == (batch_size, 1)

  if weights is not None:
    quantile_huber_loss = (batch_quantile_huber_loss * weights).mean()
  else:
    quantile_huber_loss = batch_quantile_huber_loss.mean()

  print("Quantile huber loss", quantile_huber_loss)
  return quantile_huber_loss


# note: resume here**
def evaluate_quantile_at_action(s_quantiles, actions):
  # print("Shape quantiles", s_quantiles.shape)
  assert s_quantiles.shape[0] == actions.shape[0]

  batch_size = s_quantiles.shape[0]
  N = s_quantiles.shape[1]
  # Expand actions into (batch_size, N, 1).
  action_index = actions[..., None].expand(batch_size, N, 1)

  # Calculate quantile values at specified actions.
  sa_quantiles = s_quantiles.gather(dim=2, index=action_index)

  return sa_quantiles


class RunningMeanStats:

  def __init__(self, n=10):
    self.n = n
    self.stats = deque(maxlen=n)

  def append(self, x):
    self.stats.append(x)

  def get(self):
    return np.mean(self.stats)


class LinearAnneaer:

  def __init__(self, start_value, end_value, num_steps):
    assert num_steps > 0 and isinstance(num_steps, int)

    self.steps = 0
    self.start_value = start_value
    self.end_value = end_value
    self.num_steps = num_steps

    self.a = (self.end_value - self.start_value) / self.num_steps
    self.b = self.start_value

  def step(self):
    self.steps = min(self.num_steps, self.steps + 1)

  def get(self):
    assert 0 < self.steps <= self.num_steps
    return self.a * self.steps + self.b
