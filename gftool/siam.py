"""Basic functions for the (non-interacting) single impurity Anderson model (SIAM).

The Hamiltonian for the SIAM reads

.. math::

   H = ∑_σ ϵ_σ c^†_σ c_σ + U n_↓ n_↑
     + ∑_{kσ} (V_{kσ} c^†_σ c_{kσ} + H.c.)
     + ∑_{kσ} ϵ_{kσ} n_{kσ}

The first to terms describe the interacting single impurity,
the thirds term is the hopping (or hybridization) between impurity and bath-sites,
the last term is the on-site energy of the non-interacting bath sites.

In the action formalism, the bath degrees of freedom can be readily integrated out,
as the action is quadratic in these degrees.
The local action of the impurity reads

.. math::

   S_{imp}[c^+_σ, c_σ]  = -∑_{nσ} c^+_σ [iω_n - ϵ_σ - Δ(iω_n)] c_σ + U ∫_0^β dτ n_↑(τ) n_↓(τ)

with the hybridization function

.. math:: Δ_σ(z) = ∑_{kσ} |V_{kσ}|^2 / (z - ϵ_{kσ}).

"""
import numpy as np

from gftool._util import _gu_sum
from gftool.matrix import decompose_hamiltonian
from gftool.basis.pole import (_single_pole_gf_gr_t, _single_pole_gf_le_t,
                               _single_pole_gf_ret_t)


def gf0_loc_z(z, e_onsite, e_bath, hopping_sqr):
    """Noninteracting local Green's function for the impurity.

    Parameters
    ----------
    z : (...) complex np.ndarray
        Complex frequency variable.
    e_onsite : (...) float np.ndarray
        On-site energy of the impurity site.
    e_bath : (..., Nb) float np.ndarray
        On-site energy of the bath sites.
    hopping : (..., Nb) complex np.ndarray
        Hopping matrix element between impurity and the bath sites.

    Returns
    -------
    gf0_loc_z : (...) complex np.ndarray
        Green's function of the impurity site.

    """
    return 1. / (z - e_onsite - hybrid_z(z, e_bath=e_bath, hopping_sqr=hopping_sqr))


def gf0_loc_ret_t(tt, e_onsite, e_bath, hopping):
    """Noninteracting retarded local Green's function for the impurity.

    Parameters
    ----------
    tt : (...) float np.ndarray
        Time variable. Note that the retarded Green's function is `0` for `tt<0`.
    e_onsite : (...) float np.ndarray
        On-site energy of the impurity site.
    e_bath : (..., Nb) float np.ndarray
        On-site energy of the bath sites.
    hopping : (..., Nb) complex np.ndarray
        Hopping matrix element between impurity and the bath sites.

    Returns
    -------
    gf0_loc_ret_t : (...) complex np.ndarray
        Retarded Green's function of the impurity site.

    """
    ham = hamiltonian_matrix(e_onsite, e_bath=e_bath, hopping=hopping)
    dec = decompose_hamiltonian(ham)
    # calculate only elements [..., 0] corresponding to the local impurity site
    dec.rv_inv, dec.rv = dec.rv_inv[..., :, :1], dec.rv[..., :1, :]
    eig_exp = _single_pole_gf_ret_t(tt[..., np.newaxis], dec.xi)
    gf0_t = dec.reconstruct(xi=eig_exp, kind='diag')[..., 0]
    return gf0_t


def gf0_loc_gr_t(tt, e_onsite, e_bath, hopping, beta):
    """Noninteracting greater local Green's function for the impurity.

    Parameters
    ----------
    tt : (...) float np.ndarray
        Time variable. Note that the retarded Green's function is `0` for `tt<0`.
    e_onsite : (...) float np.ndarray
        On-site energy of the impurity site.
    e_bath : (..., Nb) float np.ndarray
        On-site energy of the bath sites.
    hopping : (..., Nb) complex np.ndarray
        Hopping matrix element between impurity and the bath sites.
    beta : float
        The inverse temperature :math:`beta = 1/k_B T`.

    Returns
    -------
    gf0_loc_gr_t : (...) complex np.ndarray
        Greater Green's function of the impurity site.

    """
    ham = hamiltonian_matrix(e_onsite, e_bath=e_bath, hopping=hopping)
    dec = decompose_hamiltonian(ham)
    # calculate only elements [..., 0] corresponding to the local impurity site
    dec.rv_inv, dec.rv = dec.rv_inv[..., :, :1], dec.rv[..., :1, :]
    eig_exp = _single_pole_gf_gr_t(tt[..., np.newaxis], dec.xi, beta=beta)
    gf0_t = dec.reconstruct(xi=eig_exp, kind='diag')[..., 0]
    return gf0_t


def gf0_loc_le_t(tt, e_onsite, e_bath, hopping, beta):
    """Noninteracting lesser local Green's function for the impurity.

    Parameters
    ----------
    tt : (...) float np.ndarray
        Time variable. Note that the retarded Green's function is `0` for `tt<0`.
    e_onsite : (...) float np.ndarray
        On-site energy of the impurity site.
    e_bath : (..., Nb) float np.ndarray
        On-site energy of the bath sites.
    hopping : (..., Nb) complex np.ndarray
        Hopping matrix element between impurity and the bath sites.
    beta : float
        The inverse temperature :math:`beta = 1/k_B T`.

    Returns
    -------
    gf0_loc_le_t : (...) complex np.ndarray
        Lesser Green's function of the impurity site.

    """
    ham = hamiltonian_matrix(e_onsite, e_bath=e_bath, hopping=hopping)
    dec = decompose_hamiltonian(ham)
    # calculate only elements [..., 0] corresponding to the local impurity site
    dec.rv_inv, dec.rv = dec.rv_inv[..., :, :1], dec.rv[..., :1, :]
    eig_exp = _single_pole_gf_le_t(tt[..., np.newaxis], dec.xi, beta=beta)
    gf0_t = dec.reconstruct(xi=eig_exp, kind='diag')[..., 0]
    return gf0_t


def hamiltonian_matrix(e_onsite, e_bath, hopping):
    r"""One-particle Hamiltonian matrix of the SIAM.

    The non-interacting Hamiltonian can be written in the form

    .. math:: \hat{H} = ∑_{ijσ} c^†_{iσ} H_{ijσ} c_{jσ}.

    The Hamiltonian matrix is :math:`H_{ij}`, where we fixed the spin σ.
    The element `H_{00}` corresponds to the impurity site.

    Parameters
    ----------
    e_onsite : (...) float np.ndarray
        On-site energy of the impurity site.
    e_bath : (..., Nb) float np.ndarray
        On-site energy of the bath sites.
    hopping : (..., Nb) complex np.ndarray
        Hopping matrix element between impurity and the bath sites.
    beta : float
        The inverse temperature :math:`beta = 1/k_B T`.

    Returns
    -------
    ham_mat : (..., Nb+1, Nb+1) complex np.ndarray
        Lesser Green's function of the impurity site.

    """
    broadcast = np.broadcast(e_onsite[..., np.newaxis], e_bath, hopping)
    n_bath = broadcast.shape[-1]
    ham = np.zeros([*broadcast.shape[:-1], n_bath+1, n_bath+1], dtype=hopping.dtype)
    ham[..., 0, 0] = e_onsite
    ham[..., 0, 1:] = hopping
    ham[..., 1:, 0] = np.conj(hopping)
    ham[..., np.arange(n_bath)+1, np.arange(n_bath)+1] = e_bath
    return ham


def hybrid_z(z, e_bath, hopping_sqr):
    """Hybridization function of the impurity.

    Parameters
    ----------
    z : (...) complex np.ndarray
        Complex frequency variable.
    e_bath : (..., Nb) float np.ndarray
        On-site energy of the bath sites.
    hopping : (..., Nb) complex np.ndarray
        Hopping matrix element between impurity and the bath sites.

    Returns
    -------
    hybrid_z : (...) complex np.ndarray
        Hybridization function of the impurity site.

    """
    return _gu_sum(hopping_sqr/(np.asanyarray(z)[..., np.newaxis] - e_bath))
