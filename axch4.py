import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)

# test1: dumb log1p, when x is close to 0. sin is just a decoy.
def log1p(x):
  y = jnp.sin(x)
  return jnp.log(1. + y)

# test2: dumb log beta, when a is large and b is small
from jax.scipy.special import gammaln
def betaln(a, b):
  return gammaln(a) + gammaln(b) - gammaln(a + b)


###

import operator as op
from typing import Any, Dict, List, Callable
from jax import core
from jax._src.util import safe_map, safe_zip
from jax._src import ad_util
from jax._src import dtypes
from jax._src import source_info_util
map = safe_map
zip = safe_zip


Val = Any
Val64 = Any
Val32 = Any

def foo(jaxpr: core.ClosedJaxpr, *args):
  env: Dict[core.Var, Val64] = {}
  perturbation_env: Dict[core.Var, Val64] = {}

  def read(x: core.Atom) -> Val64:
    if isinstance(x, core.Literal):
      return x.val
    return env[x]

  def write(v: core.Var, val: Val64) -> None:
    env[v] = val

  def write_perturbation(v: core.Var, val: Val64) -> None:
    perturbation_env[v] = val

  map(write, jaxpr.jaxpr.constvars, jaxpr.consts)
  map(write, jaxpr.jaxpr.invars, args)
  for eqn in jaxpr.jaxpr.eqns:
    inputs: List[Val64] = map(read, eqn.invars)
    outputs: Union[Val64, List[Val64]] = eqn.primitive.bind(*inputs, **eqn.params)

    # TODO assumes 'params' doesn't have dtype parameters
    inputs_32: List[Val32] = map(demote_to_32, inputs)
    outputs_32: Union[Val32, List[Val32]] = eqn.primitive.bind(*inputs_32, **eqn.params)

    if eqn.primitive.multiple_results:
      outputs_32_ = map(promote_to_64, outputs_32)
      perturbations = [x - y for x, y in zip(outputs_32_, outputs)]
    else:
      outputs_32_ = promote_to_64(outputs_32)
      perturbations = outputs_32_ - outputs

    if eqn.primitive.multiple_results:
      map(write, eqn.outvars, outputs)
      map(write_perturbation, eqn.outvars, perturbations)
    else:
      write(eqn.outvars[0], outputs)
      write_perturbation(eqn.outvars[0], perturbations)

  zero_perturbations = {v: ad_util.zeros_like_aval(v.aval) for v in perturbation_env}
  sensitivity_env = jax.grad(make_perturbation_fn(jaxpr))(zero_perturbations, *args)

  # Each score is proportional to the relative error in that
  # intermediate, times the condition number of the answer with
  # respect to it.  The constant of proportionality is the final
  # value.  The value of the intermediate cancels from this computation,
  # leaving (error * derivative).
  scores = {v: jnp.vdot(perturbation_env[v] / env[v], sensitivity_env[v])
            for v in perturbation_env}
  worst_offender = max((v for eqn in jaxpr.jaxpr.eqns for v in eqn.outvars),
                       key=lambda v: jnp.abs(scores[v]))
  for worst_offender in sorted(scores, key=lambda v: jnp.abs(scores[v])):
    eqn, = (eqn for eqn in jaxpr.jaxpr.eqns if worst_offender in eqn.outvars)
    src = source_info_util.summarize(eqn.source_info)
    print(f"at {src} we applied {eqn.primitive.name} with inputs:\n" +
          '\n'.join(f'  val={read(x)}' for x in eqn.invars) + '\n' +
          f"but the output(s) had value / absolute / relative error:\n" +
          '\n'.join(f'  {env[v]} / {perturbation_env[v]} / {perturbation_env[v] / env[v]}'
                    for v in eqn.outvars) + '\n' +
          f"with an output sensitivity of {sensitivity_env[worst_offender]};\n" +
          f"this resulted in an elasticity score of {scores[worst_offender]}\n"
          )

x64_to_x32 = {
    jnp.dtype('float64'): jnp.dtype('float32')
}
x32_to_x64 = {v:k for k, v in x64_to_x32.items()}

def demote_to_32(x):
  new_dtype = x64_to_x32[dtypes.dtype(x)]
  return jax.lax.convert_element_type(x, new_dtype)

def promote_to_64(x):
  new_dtype = x32_to_x64[dtypes.dtype(x)]
  return jax.lax.convert_element_type(x, new_dtype)


def make_perturbation_fn(jaxpr: core.ClosedJaxpr) -> Callable:
  def fn(perturbation_env, *args):
    env = {}

    def read(x: core.Atom) -> Val:
      return env[x] if isinstance(x, core.Var) else x.val

    def read_perturbation(v: core.Var) -> Val:
      return perturbation_env[v]

    def write(v: core.Var, val: Val) -> None:
      env[v] = val

    map(write, jaxpr.jaxpr.constvars, jaxpr.consts)
    map(write, jaxpr.jaxpr.invars, args)
    for eqn in jaxpr.jaxpr.eqns:
      inputs = map(read, eqn.invars)
      outputs = eqn.primitive.bind(*inputs, **eqn.params)
      perturbations = map(read_perturbation, eqn.outvars)
      if eqn.primitive.multiple_results:
        outputs = map(op.add, outputs, perturbations)
        map(write, eqn.outvars, outputs)
      else:
        outputs = outputs + read_perturbation(eqn.outvars[0])
        write(eqn.outvars[0], outputs)

    out, = map(read, jaxpr.jaxpr.outvars)
    return out
  return fn


# x = 1e-4
# jaxpr = jax.make_jaxpr(log1p)(x)
# foo(jaxpr, x)

# TODO recurse into higher-order primitives

def exp_gamma_log_prob(concentration, log_rate, x, foo, bar):
  y = jnp.exp(x + log_rate)
  log_unnormalized_prob = concentration * x - y
  # log_unnormalized_prob = (1e1 + log_unnormalized_prob_) - 1e1
  log_normalization = jax.lax.lgamma(concentration) - concentration * log_rate
  ans = log_unnormalized_prob - log_normalization
  quux = foo - bar
  return ans + quux

conc = 117.67729
log_rate = 159.94534
x = -155.34862
frob = 1400.0
bar = 1399.1

jaxpr = jax.make_jaxpr(exp_gamma_log_prob)(conc, log_rate, x, frob, bar)
foo(jaxpr, conc, log_rate, x, frob, bar)


def norm_pytree(stuff):
  stuff_v, _ = ravel_pytree(stuff)
  print(stuff_v, type(stuff_v))
  return jnp.linalg.norm(stuff_v)

def condition_number(f, *args, **kwargs):
  fx, grads = jax.value_and_grad(f, argnums=range(len(args)))(*args, **kwargs)
  return norm_pytree(args) * norm_pytree(grads) / jnp.abs(fx)

# print(condition_number(exp_gamma_log_prob, 117.67729, 159.94534, -155.34862))

def exp_gamma_log_prob_offset(concentration, x_offset):
  """Semantically equal to
    exp_gamma_log_prob(concentration, log_rate, x_offset - log_rate)
  but computed more accurately.
  This is useful when you want
    exp_gamma_log_prob(concentration, log_rate, x)
  but it's ill-conditioned because x ~= -log_rate.
  If, in addition, you have access to
    x_offset = x + log_rate
  more accurately than just computing that, then you can use this function.
  """
  y = jnp.exp(x_offset)
  log_unnormalized_prob_offset = concentration * x_offset - y
  log_normalization_offset = jax.lax.lgamma(concentration)
  return log_unnormalized_prob_offset - log_normalization_offset

offset = x + log_rate

jaxpr2 = jax.make_jaxpr(exp_gamma_log_prob_offset)(conc, offset)
# foo(jaxpr2, conc, offset)

print(condition_number(exp_gamma_log_prob_offset, conc, offset))

# Experience report

# The `foo` tool is useful.  I already knew that the problem was a
# catastrophic cancellation between log_unnormalized_prob and
# log_normalization, but the tool clued me in that the cancellation in
# x + log_prob was also important -- it induced small relative error
# in the intermediate value, but had a relatively high elasticity
# score (by virtue of passing through `exp`) (though still 2 orders of
# magnitude lower than the catastrophic cancellation).  This led me to
# try `exp_gamma_log_prob_offset`, pushing the computation of x +
# log_prob to the client in hopes that they may be able to do it more
# accurately due to knowing something about where x and log_rate came
# from.  The result is both significantly better conditioned and
# computable far more accurately.  However, it is still less than
# perfect on both counts, because there is still a catastrophic
# cancellation between log_unnormalized_prob_offset and
# log_normalization_offset.  It's just ~2 orders of magnitude less
# catastrophic than before (on these inputs).

# Can't tell what the right formula is.  We tried
#   score(v) = <relative error at v> * <d ans / dv>
# and it seemed to do well on this problem.  It seems more sensible
# to use
#   score2(v) = <relative error at v> * <condition number of ans wrt v>.
#
# The relationship between these is that
#   score2(v) = v * score(v) / ans.
# Since ans doesn't vary, dividing by it has no effect on the ordering
# of the v by their scores, so we're just multiying by the value of
# the intermediate.
#
# I tried score2, but it performed worse than score: it painted all
# the additive terms at the end; it seems like it was saying "there is
# a problem downstream of this that this intermediate contributes to",
# instead of pegging the location of said problem.  In particular,
# those terms all had small relative error and high score2, because
# they were themselves large.
#
# Now that I think about it, we also have
#   score2(v) = <absolute error at v> * <d ans / dv> / ans.
