


from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCDiscreteActor
from utils.networks_actionless import (
    VelocityEncoder,
    VelocityCritic,
    ActionVelocityMap,
    VCRLActor,
)


class VCRLAgent(flax.struct.PyTreeNode):
    """Contrastive RL (CRL) agent.

    This implementation supports both AWR (actor_loss='awr') and DDPG+BC (actor_loss='ddpgbc') for the actor loss.
    CRL with DDPG+BC only fits a Q function, while CRL with AWR fits both Q and V functions to compute advantages.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def contrastive_loss(self, batch, velocity_encodings, grad_params):
        """Compute the contrastive value loss for the Q or V function."""
        batch_size = batch['observations'].shape[0]

        v, phi, psi = self.network.select('critic')(
            batch['observations'],
            velocity_encodings,
            batch['value_goals'],
            info=True,
            params=grad_params,
        )
        if len(phi.shape) == 2:  # Non-ensemble.
            phi = phi[None, ...]
            psi = psi[None, ...]

        logits = jnp.einsum('eik,ejk->ije', phi, psi) / jnp.sqrt(phi.shape[-1])

        # logits.shape is (B, B, e) with one term for positive pair and (B - 1) terms for negative pairs in each row.
        I = jnp.eye(batch_size)
        contrastive_loss = jax.vmap(
            lambda _logits: optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I),
            in_axes=-1,
            out_axes=-1,
        )(logits)
        contrastive_loss = jnp.mean(contrastive_loss)

        # Compute additional statistics.
        v = jnp.exp(v)
        logits = jnp.mean(logits, axis=-1)
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return contrastive_loss, {
            'contrastive_loss': contrastive_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'binary_accuracy': jnp.mean((logits > 0) == I),
            'categorical_accuracy': jnp.mean(correct),
            'logits_pos': logits_pos,
            'logits_neg': logits_neg,
            'logits': logits.mean(),
        }


    def actor_loss(self, batch, grad_params):
        batch_size = batch['observations'].shape[0]
        # For all of the observations and goals in the batch, use actor to get actions, 
        # use action velocity map to get corresponding velocity encodings,
        # and pass those to critic and maximize.
        actions = jnp.clip(
            self.network.select('actor')(
                batch['observations'],
                batch['actor_goals'],
                temperature=1.0,
                params=grad_params,
            ).mode(),
            -1,
            1,
        )
        assert actions.shape[0] == batch_size
        # Do not update the action velocity map or the critic in this loss
        velocity_encodings = self.network.select('action_velocity_map')(
            actions,
            observations=batch['observations'],
        )
        q1, q2 = self.network.select('critic')(
            batch['observations'],
            velocity_encodings,
            batch['actor_goals'],
        )
        assert q1.shape == q2.shape
        q = jnp.minimum(q1, q2)
        actor_loss_unnormalized = -jnp.mean(q)
        actor_loss = actor_loss_unnormalized / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
        return actor_loss, {
            'actor_loss_unnorm': actor_loss_unnormalized,
        }

    def action_velocity_map_loss(self, batch, velocity_encodings, grad_params):
        # Batch is actionful
        velocity_encodings_predicted = self.network.select('action_velocity_map')(
            batch['actions'],
            observations=batch['observations'],
            params=grad_params,
        )

        # Compute MSE loss between predicted and actual velocity encodings
        loss = jnp.mean((velocity_encodings_predicted - jax.lax.stop_gradient(velocity_encodings)) ** 2)

        return loss, {
            'action_velocity_map_loss': loss,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        combined_batch = {
            key: jnp.concatenate([batch["actionful"][key], batch["actionless"][key]], axis=0)
            for key in batch["actionful"].keys()
            if key != 'actions'
        }
        combined_batch_size = combined_batch['observations'].shape[0]

        # Get velocity encodings for all transitions
        velocity_encodings = self.network.select('velocity_encoder')(
            combined_batch['observations'],
            combined_batch['next_observations'],
            params=grad_params,
        )
        assert velocity_encodings.shape == (combined_batch_size, self.config['velocity_encoding_dim'])
        velocity_encodings_actionful = velocity_encodings[: batch["actionful"]['observations'].shape[0]]

        # Compute contrastive critic loss with velocity encodings for all data
        critic_loss, critic_info = self.contrastive_loss(combined_batch, velocity_encodings, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        # Compute action velocity map loss on actionful data
        action_velocity_map_loss, action_velocity_map_info = self.action_velocity_map_loss(
            batch["actionful"],
            velocity_encodings_actionful,
            grad_params,
        )
        for k, v in action_velocity_map_info.items():
            info[f'action_velocity_map/{k}'] = v

        # Compute actor loss on all data
        actor_loss, actor_info = self.actor_loss(combined_batch, grad_params)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + action_velocity_map_loss + actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        assert not config['discrete']
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_next_observations = ex_goals = ex_observations
        action_dim = ex_actions.shape[-1]

        print("Action dim:", action_dim, " State dim:", ex_observations.shape[-1])

        velocity_encoder_def = VelocityEncoder(
            hidden_dims=config['velocity_encoder_hidden_dims'],
            layer_norm=config['layer_norm'],
            velocity_encoding_dim=config['velocity_encoding_dim'],
        )

        # Pre-initialize velocity encoder to get example velocity encoding
        velocity_encoder_rng, init_rng = jax.random.split(init_rng)
        velocity_encoder_params = velocity_encoder_def.init(velocity_encoder_rng, ex_observations, ex_next_observations)['params']
        ex_velocity_encodings = velocity_encoder_def.apply({'params': velocity_encoder_params}, ex_observations, ex_next_observations)

        # Define critic network
        critic_def = VelocityCritic(
            hidden_dims=config['critic_hidden_dims'],
            latent_dim=config['latent_dim'],
            layer_norm=config['layer_norm'],
        )

        # Define action velocity map
        action_velocity_map_def = ActionVelocityMap(
            hidden_dims=config['action_velocity_map_hidden_dims'],
            velocity_encoding_dim=config['velocity_encoding_dim'],
            layer_norm=config['layer_norm'],
        )

        # Define actor network.
        actor_def = VCRLActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
        )

        # Set up network info with all networks.
        network_info = dict(
            velocity_encoder=(velocity_encoder_def, (ex_observations, ex_next_observations)),
            critic=(critic_def, (ex_observations, ex_velocity_encodings, ex_goals)),
            action_velocity_map=(action_velocity_map_def, (ex_actions, ex_observations)),
            actor=(actor_def, (ex_observations, ex_goals))
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='vcrl',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            # Velocity encoder
            velocity_encoder_hidden_dims=(128,),
            velocity_encoding_dim=2,
            # Critic
            critic_hidden_dims=(512, 512, 512),
            latent_dim=512,  # Latent dimension for phi and psi.
            # Action velocity map
            action_velocity_map_hidden_dims=(128,),
            # Actor
            actor_hidden_dims=(512, 512, 512),

            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            actor_loss='', # not supported

            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,

            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.0,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=False,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
