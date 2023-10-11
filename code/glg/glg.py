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
    def __init__(self, num_seasons=None, num_season_weeks=36, xmas_half_window=2, transform=None):
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
        self.num_season_weeks = num_season_weeks
        
        if transform is not None and \
            (transform not in ['4rt', 'sqrt']):
            raise ValueError('transform must be None, "4rt", or "sqrt".')
        self.transform = transform
        
        self.xmas_half_window = xmas_half_window
        self.xmas_window = 2 * xmas_half_window + 1
    
    
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
    
    
    def make_ar1_precision(self, dim, rho, sigma):
        result = jnp.identity(dim) \
                 + jnp.diag(jnp.concatenate([jnp.zeros((1,)),
                                             jnp.full((dim-2,), jnp.power(rho, 2)),
                                             jnp.zeros((1,))],
                                             axis=0)) \
                 - jnp.diag(jnp.repeat(rho, dim-1), k=-1) \
                 - jnp.diag(jnp.repeat(rho, dim-1), k=1)
        result = result / jnp.power(sigma, 2)
        return result
    
    
    def xmas_effect(self, s, w, w_xmas, xmas_mean_effect, xmas_season_effect):
        xmas_effects_expanded = jnp.zeros((self.num_seasons, self.num_season_weeks))
        col_inds = w_xmas.reshape(self.num_seasons, 1) + \
                   jnp.arange(-self.xmas_half_window, self.xmas_half_window + 1)
        col_inds = col_inds.flatten()
        row_inds = jnp.repeat(jnp.arange(self.num_seasons), self.xmas_window)
        xmas_effects_expanded = xmas_effects_expanded \
            .at[row_inds, col_inds] \
            .set((xmas_mean_effect + xmas_season_effect).flatten())
        return xmas_effects_expanded.at[s, w].get()
    
    
    def model(self,
              y_trans_0=None, s_0=None, w_0=None,
              y_trans_1=None, s_1=None, w_1=None,
              w_xmas=None):
        '''
        Generalized logistic growth model for disease incidence
        
        Parameters
        ----------
        y_trans: array with shape (num_obs,)
            Incidence on a transformed scale
        s: array with shape (num_obs,)
            Season value for each observation, from 0 to self.num_seasons - 1
        w: array with shape (num_obs,)
            Season week for each observation
        w_xmas: array with shape (self.num_seasons,)
            Season week in which Christmas occurred for each season
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
        delta_concentration_0 = numpyro.sample(
            'delta_concentration_0',
            dist.Exponential(rate=10))
        delta_rate_0 = numpyro.sample(
            'delta_rate_0',
            dist.Exponential(rate=10))
        delta_0 = numpyro.sample(
            'delta_0',
            dist.Gamma(concentration=delta_concentration_0,
                       rate=delta_rate_0),
            sample_shape=(self.num_seasons,))
        
        ## beta = growth rate parameter: B in Wikipedia's notation
        beta_concentration_0 = numpyro.sample(
            'beta_concentration_0',
            dist.Exponential(rate=10))
        beta_rate_0 = numpyro.sample(
            'beta_rate_0',
            dist.Exponential(rate=10))
        beta_0 = numpyro.sample(
            'beta_0',
            dist.Gamma(concentration=beta_concentration_0,
                       rate=beta_rate_0),
            sample_shape=(self.num_seasons,))
        
        ## reference week parameter: M
        ref_w_mean_0 = numpyro.sample(
            'ref_w_mean_0',
            dist.Normal(loc=26, scale = 10))
        ref_w_scale_0 = numpyro.sample(
            'ref_w_scale_0',
            dist.HalfNormal(10))
        ref_w_0 = numpyro.sample(
            'ref_w_0',
            dist.Normal(loc=ref_w_mean_0, scale=ref_w_scale_0),
            sample_shape=(self.num_seasons,))
        
        ## which side of peak experiences faster growth: nu
        nu_concentration_0 = numpyro.sample(
            'nu_concentration_0',
            dist.Exponential(rate=10))
        nu_rate_0 = numpyro.sample(
            'nu_rate_0',
            dist.Exponential(rate=10))
        nu_0 = numpyro.sample(
            'nu_0',
            dist.Gamma(concentration=nu_concentration_0,
                       rate=nu_rate_0),
            sample_shape=(self.num_seasons,))
        
        
        # Christmas/holiday effect parameters
        xmas_mean_ar_rho_0 = numpyro.sample(
            'xmas_mean_ar_rho_0',
            dist.Uniform()
        )
        xmas_mean_ar_sigma_0 = numpyro.sample(
            'xmas_mean_ar_sigma_0',
            dist.HalfNormal()
        )
        xmas_mean_effect_0 = numpyro.sample(
            'xmas_mean_effect_0',
            dist.MultivariateNormal(precision_matrix=self.make_ar1_precision(dim=self.xmas_window,
                                                                             rho=xmas_mean_ar_rho_0,
                                                                             sigma=xmas_mean_ar_sigma_0))
        )

        xmas_season_ar_rho_0 = numpyro.sample(
            'xmas_season_ar_rho_0',
            dist.Uniform()
        )
        xmas_season_ar_sigma_0 = numpyro.sample(
            'xmas_season_ar_sigma_0',
            dist.HalfNormal()
        )
        xmas_season_effect_0 = numpyro.sample(
            'xmas_season_effect_0',
            dist.MultivariateNormal(
                precision_matrix=self.make_ar1_precision(dim=self.xmas_window,
                                                         rho=xmas_season_ar_rho_0,
                                                         sigma=xmas_season_ar_sigma_0)),
            sample_shape=(self.num_seasons,)
        )
        # print('xmas_season_effect.shape')
        # print(xmas_season_effect.shape)
        
        ## delta = upper asymptote minus lower asymptote: (K - A) in Wikipedia's notation
        # xmas_delta_concentration = numpyro.sample(
        #     'xmas_delta_concentration',
        #     dist.Exponential(rate=10))
        # xmas_delta_rate = numpyro.sample(
        #     'xmas_delta_rate',
        #     dist.Exponential(rate=10))
        # xmas_delta = numpyro.sample(
        #     'xmas_delta',
        #     dist.Gamma(concentration=xmas_delta_concentration,
        #                rate=xmas_delta_rate),
        #     sample_shape=(self.num_seasons,))
        
        # ## beta = growth rate parameter: B in Wikipedia's notation
        # xmas_beta_concentration = numpyro.sample(
        #     'xmas_beta_concentration',
        #     dist.Exponential(rate=10))
        # xmas_beta_rate = numpyro.sample(
        #     'xmas_beta_rate',
        #     dist.Exponential(rate=10))
        # xmas_beta = numpyro.sample(
        #     'xmas_beta',
        #     dist.Gamma(concentration=xmas_beta_concentration,
        #                rate=xmas_beta_rate),
        #     sample_shape=(self.num_seasons,))
        
        # ## reference week parameter: M
        # ## Note: for christmas, we will say this is the week of Christmas plus a (small) offset
        # xmas_ref_w_mean_offset = numpyro.sample(
        #     'xmas_ref_w_mean_offset',
        #     dist.Normal(loc=0, scale = 1))
        # xmas_ref_w_scale = numpyro.sample(
        #     'xmas_ref_w_scale',
        #     dist.HalfNormal(1))
        # xmas_ref_w = numpyro.sample(
        #     'xmas_ref_w',
        #     dist.Normal(loc=w_xmas + xmas_ref_w_mean_offset, scale=xmas_ref_w_scale))
        
        # ## which side of peak experiences faster growth: nu
        # xmas_nu_concentration = numpyro.sample(
        #     'xmas_nu_concentration',
        #     dist.Exponential(rate=10))
        # xmas_nu_rate = numpyro.sample(
        #     'xmas_nu_rate',
        #     dist.Exponential(rate=10))
        # xmas_nu = numpyro.sample(
        #     'xmas_nu',
        #     dist.Gamma(concentration=xmas_nu_concentration,
        #                rate=xmas_nu_rate),
        #     sample_shape=(self.num_seasons,))
        
        # mean (on transformed scale)
        ## sum of main curve and Christmas/holiday curve 
        # print('delta.shape')
        # print(delta.shape)
        # print('beta.shape')
        # print(beta.shape)
        # print('ref_w.shape')
        # print(ref_w.shape)
        # print('nu.shape')
        # print(nu.shape)
        # print('xmas_delta.shape')
        # print(xmas_delta.shape)
        # print('xmas_beta.shape')
        # print(xmas_beta.shape)
        # print('xmas_ref_w.shape')
        # print(xmas_ref_w.shape)
        # print('xmas_nu.shape')
        # print(xmas_nu.shape)
        mean_y_0 = self.glg_inc_curve(s_0, w_0, delta_0, beta_0, ref_w_0, nu_0)
        
        # for xmas_offset in jnp.arange(-self.xmas_half_window, self.xmas_half_window + 1):
        #     w_hol = w_xmas + xmas_offset
        
        if self.transform is None:
            mean_y_trans_0 = mean_y_0 + \
                self.xmas_effect(s_0, w_0, w_xmas, xmas_mean_effect_0, xmas_season_effect_0)
        elif self.transform == '4rt':
            mean_y_trans_0 = jnp.power(0.01 + mean_y_0, 0.25) + \
                self.xmas_effect(s_0, w_0, w_xmas, xmas_mean_effect_0, xmas_season_effect_0)
        elif self.transform == 'sqrt':
            mean_y_trans_0 = jnp.sqrt(0.01 + mean_y_0) + \
                self.xmas_effect(s_0, w_0, w_xmas, xmas_mean_effect_0, xmas_season_effect_0)
        
        # standard deviation of observation noise on transformed scale
        sigma_0 = numpyro.sample('sigma', dist.HalfNormal(0.1))
        
        # observation model for y on transformed scale
        numpyro.sample(
            'y_trans_0',
            dist.Normal(loc=mean_y_trans_0, scale=sigma_0),
            obs=y_trans_0)
    
    
    def fit(self, y_0, s_0, w_0, y_1, s_1, w_1, w_xmas,
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
        w_xmas: array with shape (self.num_seasons,)
            Season week in which Christmas occurred for each season
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
            y_trans_0 = y_0
            y_trans_1 = y_1
        elif self.transform == 'log':
            y_trans_0 = jnp.log(y_0)
            y_trans_1 = jnp.log(y_1)
        elif self.transform == '4rt':
            y_trans_0 = jnp.power(0.01 + y_0, 0.25)
            y_trans_1 = jnp.power(0.01 + y_1, 0.25)
        elif self.transform == 'sqrt':
            y_trans_0 = jnp.sqrt(0.01 + y_0)
            y_trans_1 = jnp.sqrt(0.01 + y_1)
        
        self.mcmc.run(rng_key,
                      y_trans_0=y_trans_0, s_0=s_0, w_0=w_0,
                      y_trans_1=y_trans_1, s_1=s_1, w_1=w_1,
                      w_xmas=w_xmas)
        print('\nMCMC elapsed time:', time.time() - start)
        
        if print_summary:
            self.mcmc.print_summary()
        return self.mcmc.get_samples()
    
    
    def sample(self, rng_key, s, w, w_xmas,
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
        
        preds = predictive(rng_key, s_0=s, w_0=w, s_1=s, w_1=w, w_xmas=w_xmas)
        
        if self.transform is None:
            preds['y_0'] = preds['y_trans_0']
            preds['y_1'] = preds['y_trans_1']
        elif self.transform == 'log':
            preds['y_0'] = jnp.exp(preds['y_trans_0'])
            preds['y_1'] = jnp.exp(preds['y_trans_1'])
        elif self.transform == '4rt':
            preds['y_0'] = jnp.power(preds['y_trans_0'], 4)
            preds['y_1'] = jnp.power(preds['y_trans_1'], 4)
        elif self.transform == 'sqrt':
            preds['y_0'] = jnp.power(preds['y_trans_0'], 2)
            preds['y_1'] = jnp.power(preds['y_trans_1'], 2)
        
        return preds
