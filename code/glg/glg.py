import os
import time

import jax.numpy as jnp

import numpyro
from numpyro.contrib.control_flow import scan
import numpyro.distributions as dist


class GLG():
    '''
    Class for a hierarchical generalized logistic growth model.
    '''
    def __init__(self, num_seasons=None, transform=None):
        '''
        Initialize a GLG model
        
        Parameters
        ----------
        num_seasons: integer or None
            Number of seasons. If None, will be set at the time of fitting
        transform: string or None
            Data transformation to use. None means no transformation; other
            options are 'log', '4rt', and 'sqrt'
        
        Returns
        -------
        None
        '''
        if num_seasons is not None and \
                (type(num_seasons) is not int or num_seasons <= 0):
            raise ValueError('num_seasons must be None or a positive integer')
        self.num_seasons = num_seasons
        
        if transform is not None and \
            (transform not in ['log', '4rt', 'sqrt']):
            raise ValueError('transform must be None or one of "log", "4rt", or "sqrt".')
        self.transform = transform
    
    
    def glg_inc_curve(self, s, w, delta, beta, ref_w, nu):
        '''
        generalized logistic growth model
        See Wikipedia: https://en.wikipedia.org/wiki/Generalised_logistic_function
        Cumulative values parameterized as
        Y(t) = A + delta * [1 + exp(-beta * (w - ref_w))]^(-1/nu)
        Taking a derivative, we obtain the following model for incidence:
        y(t) = -(delta / nu) * [1 + exp(-beta * (w - ref_w))]^(-1/nu - 1) * exp(-beta * (w - ref_w)) * (-beta)
             = (beta * delta / nu) * exp(-beta * (w - ref_w)) * [1 + exp(-beta * (w - ref_w))]^(-1/nu - 1)
        https://www.wolframalpha.com/input?i=d%2Fdt+a+%2B+%28k+-+a%29+%2F+%281+%2B+q*exp%28-b*t%29%29%5E%281%2Fv%29
        '''
        exp_term = jnp.exp(-beta[s] * (w - ref_w[s]))
        glg_inc = beta[s] * delta[s] / nu[s] \
                  * exp_term * jnp.power(1.0 + exp_term, -1.0 / nu[s] - 1.0)
        return glg_inc
    
    
    def model(self, y_trans=None, s=None, w=None):
        '''
        Shooting proportion model with offensive player and defensive team effects.
        
        Parameters
        ----------
        y_trans: array with shape (num_obs,)
            Incidence on a transformed scale
        s: array with shape (num_obs,)
            Season value for each observation, from 0 to self.num_seasons - 1
        w: array with shape (num_obs,)
            Season week for each observation
        '''
        # # acquire and/or validate number of time steps and series
        # if y is not None:
        #     if self.num_timesteps is not None and self.num_timesteps != y.shape[0]:
        #         raise ValueError('if provided, require num_timesteps = y.shape[0]')
        #     if self.num_players is not None and self.num_players != y.shape[1]:
        #         raise ValueError('if provided, require num_series = y.shape[1]')
        #     self.num_timesteps, self.num_players = y.shape
        
        # if self.num_timesteps is None or self.num_players is None:
        #     raise ValueError('Must provide either y or both of num_timesteps and num_players')
        
        # main curve parameters
        ## delta = upper asymptote minus lower asymptote: (K - A) in Wikipedia's notation
        delta_concentration = numpyro.sample(
            'delta_concentration',
            dist.Exponential(rate=10))
        delta_rate = numpyro.sample(
            'delta_rate',
            dist.Exponential(rate=10))
        delta = numpyro.sample(
            'delta',
            dist.Gamma(concentration=delta_concentration,
                       rate=delta_rate),
            sample_shape=(self.num_seasons,))
        
        
        # beta = growth rate parameter: B in Wikipedia's notation
        beta_concentration = numpyro.sample(
            'beta_concentration',
            dist.Exponential(rate=10))
        beta_rate = numpyro.sample(
            'beta_rate',
            dist.Exponential(rate=10))
        beta = numpyro.sample(
            'beta',
            dist.Gamma(concentration=beta_concentration,
                       rate=beta_rate),
            sample_shape=(self.num_seasons,))
        
        # reference week parameter: M
        ref_w_mean = numpyro.sample(
            'ref_w_mean',
            dist.Normal(loc=26, scale = 10))
        ref_w_scale = numpyro.sample(
            'ref_w_scale',
            dist.HalfNormal(10))
        ref_w = numpyro.sample(
            'ref_w',
            dist.Normal(loc=ref_w_mean, scale=ref_w_scale),
            sample_shape=(self.num_seasons,))
        
        # which side of peak experiences faster growth: nu
        nu_concentration = numpyro.sample(
            'nu_concentration',
            dist.Exponential(rate=10))
        nu_rate = numpyro.sample(
            'nu_rate',
            dist.Exponential(rate=10))
        nu = numpyro.sample(
            'nu',
            dist.Gamma(concentration=nu_concentration,
                       rate=nu_rate),
            sample_shape=(self.num_seasons,))
        
        if self.transform is None:
            mean_y_trans = self.glg_inc_curve(s, w, delta, beta, ref_w, nu)
        elif self.transform == '4rt':
            mean_y_trans = jnp.power(0.01 + self.glg_inc_curve(s, w, delta, beta, ref_w, nu), 0.25)
        elif self.transform == 'sqrt':
            mean_y_trans = jnp.sqrt(0.01 + self.glg_inc_curve(s, w, delta, beta, ref_w, nu))
        
        # standard deviation of observation noise on transformed scale
        sigma = numpyro.sample('sigma', dist.HalfNormal(1))
        
        # observation model for y
        numpyro.sample(
            'y_trans',
            dist.Normal(loc=mean_y_trans, scale=sigma),
            obs=y_trans)
    
    
    def fit(self, y, s, w,
            rng_key, num_warmup=1000, num_samples=1000, num_chains=1,
            print_summary=False):
        '''
        Fit model using MCMC
        
        Parameters
        ----------
        y: array with shape (num_obs,)
            Incidence
        s: array with shape (num_obs,)
            Season value for each observation, from 0 to self.num_seasons - 1
        w: array with shape (num_obs,)
            Season week for each observation
        rng_key: random.PRNGKey
            Random number generator key to be used for MCMC sampling
        num_warmup: integer
            Number of warmup steps for the MCMC algorithm
        num_samples: integer
            Number of sampling steps for the MCMC algorithm
        num_chains: integer
            Number of MCMC chains to run
        print_summary: boolean
            If True, print a summary of estimation results
        
        Returns
        -------
        array with samples from the posterior distribution of the model parameters
        '''
        start = time.time()
        sampler = numpyro.infer.NUTS(self.model)
        self.mcmc = numpyro.infer.MCMC(
            sampler,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=False if 'NUMPYRO_SPHINXBUILD' in os.environ else True)
        
        if self.transform is None:
            y_trans = y
        elif self.transform == 'log':
            y_trans = jnp.log(y)
        elif self.transform == '4rt':
            y_trans = jnp.power(0.01 + y, 0.25)
        elif self.transform == 'sqrt':
            y_trans = jnp.sqrt(0.01 + y)
        
        self.mcmc.run(rng_key, y_trans=y_trans, s=s, w=w)
        print('\nMCMC elapsed time:', time.time() - start)
        
        if print_summary:
            self.mcmc.print_summary()
        return self.mcmc.get_samples()
    
    
    def sample(self, rng_key, s, w,
               condition={}, num_samples=1):
        '''
        Draw a sample from the joint distribution of parameter values and data
        defined by the model, possibly conditioning on a set of fixed values.
        
        Parameters
        ----------
        rng_key: random.PRNGKey
            Random number generator key to be used for sampling
        condition: dictionary
            Optional dictionary of parameter values to hold fixed
        num_samples: integer
            Number of samples to draw. Ignored if condition is provided, in
            which case the number of samples will correspond to the shape of
            the entries in condition.
        
        Returns
        -------
        dictionary of arrays of sampled values
        '''
        if condition == {}:
            predictive = numpyro.infer.Predictive(self.model,
                                                  num_samples=num_samples)
        else:
            predictive = numpyro.infer.Predictive(self.model,
                                                  posterior_samples=condition)
        
        preds = predictive(rng_key, s=s, w=w)
        
        if self.transform is None:
            preds['y'] = preds['y_trans']
        elif self.transform == 'log':
            preds['y'] = jnp.exp(preds['y_trans'])
        elif self.transform == '4rt':
            preds['y'] = jnp.power(preds['y_trans'], 4)
        elif self.transform == 'sqrt':
            preds['y'] = jnp.power(preds['y_trans'], 2)
        
        return preds
