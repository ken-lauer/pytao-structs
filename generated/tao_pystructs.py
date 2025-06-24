#!/usr/bin/env python
# vi: syntax=python sw=4 ts=4 sts=4
"""
This file is auto-generated; do not hand-edit it.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import textwrap
from typing import (
    Annotated,
    Any,
    ClassVar,
    Sequence,
    cast,
)

import numpy as np
import pydantic
from pytao import Tao
from rich.pretty import pretty_repr
from typing_extensions import Self

logger = logging.getLogger(__name__)


def _sequence_helper(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (int, float)):
        return [value]
    return list(value)


def _check_equality(obj1: Any, obj2: Any) -> bool:
    """
    Check equality of `obj1` and `obj2`.`

    Parameters
    ----------
    obj1 : Any
    obj2 : Any

    Returns
    -------
    bool
    """
    # TODO: borrowed from lume-genesis; put this in a reusable spot
    if not isinstance(obj1, type(obj2)):
        return False

    if isinstance(obj1, pydantic.BaseModel):
        return all(
            _check_equality(
                getattr(obj1, attr),
                getattr(obj2, attr),
            )
            for attr, fld in obj1.model_fields.items()
            if not fld.exclude
        )

    if isinstance(obj1, dict):
        if set(obj1) != set(obj2):
            return False

        return all(
            _check_equality(
                obj1[key],
                obj2[key],
            )
            for key in obj1
        )

    if isinstance(obj1, (list, tuple)):
        if len(obj1) != len(obj2):
            return False
        return all(
            _check_equality(obj1_value, obj2_value) for obj1_value, obj2_value in zip(obj1, obj2)
        )

    if isinstance(obj1, np.ndarray):
        if not obj1.shape and not obj2.shape:
            return True
        return np.allclose(obj1, obj2)

    if isinstance(obj1, float):
        return np.allclose(obj1, obj2)

    return bool(obj1 == obj2)


FloatSequence = Annotated[Sequence[float], pydantic.BeforeValidator(_sequence_helper)]
IntSequence = Annotated[Sequence[int], pydantic.BeforeValidator(_sequence_helper)]
ArgumentType = int | float | str | IntSequence | FloatSequence


class TaoModel(
    pydantic.BaseModel,
    str_strip_whitespace=True,  # Strip whitespace from strings
    str_min_length=0,  # We can't write empty strings currently
    validate_assignment=True,
    extra="forbid",
):
    """
    A helper base class which allows for creating/updating an instance with Tao objects.
    """

    _tao_command_: ClassVar[str]
    _tao_command_default_args_: ClassVar[dict[str, Any]]

    command_args: dict[str, ArgumentType] = pydantic.Field(
        default_factory=dict,
        frozen=True,
        description="Arguments used for the pytao command to generate this structure",
        repr=False,
    )

    def query(self, tao: Tao) -> Self:
        """Query Tao again to generate a new instance of this model."""
        return self.from_tao(tao, **self.command_args)

    @classmethod
    def from_tao(cls: type[Self], tao: Tao, **kwargs) -> Self:
        """
        Create this structure by querying Tao for its current values.

        Parameters
        ----------
        tao : Tao
        **kwargs :
            Keyword arguments to pass to the relevant ``tao`` command.
        """
        cmd_kwargs = dict(cls._tao_command_default_args_)
        cmd_kwargs.update(**kwargs)

        cmd = getattr(tao, cls._tao_command_)
        data = cmd(**cmd_kwargs)
        return cls(command_args=cmd_kwargs, **data)

    def __eq__(self, other) -> bool:
        return _check_equality(self, other)

    def __repr__(self):
        return pretty_repr(self)


class TaoSettableModel(TaoModel):
    """
    A helper base class which allows for setting Tao parameters based on
    instance attributes.
    """

    # Do not set these keys if the values are 0, avoiding setting other things.
    _tao_skip_if_0_: ClassVar[tuple[str, ...]]

    @property
    def settable_fields(self) -> list[str]:
        """Names of all 'settable' (modifiable) fields."""
        return [attr for attr, field_info in self.model_fields.items() if not field_info.frozen]

    @property
    def _all_attributes_to_set(self):
        for attr in self.settable_fields:
            value = getattr(self, attr)

            if attr in self._tao_skip_if_0_ and value == 0:
                continue

            if np.isscalar(value):
                yield attr, None, value
            else:
                for index, val in enumerate(value):
                    yield attr, index, val

    def _get_changed_attributes(self, tao: Tao):
        current = self.query(tao)

        cmds = []
        for attr, index, value in self._all_attributes_to_set:
            current_value = getattr(current, attr)
            new_value = getattr(self, attr)
            if index is not None:
                new_value = new_value[index]
                current_value = current_value[index]

            if not _check_equality(current_value, new_value):
                yield attr, index, value

        return cmds

    def get_set_commands(self, tao: Tao | None = None):
        """
        Generate a list of set commands for attributes.

        Parameters
        ----------
        tao : Tao or None, optional
            An instance of the Tao class, if provided. If `None`, all attributes
            to be set will be used.

        Returns
        -------
        cmds : list of str
        """
        cmds = []
        if tao is not None:
            attrs = self._get_changed_attributes(tao)
        else:
            attrs = self._all_attributes_to_set

        for attr, index, value in attrs:
            if index is None:
                cmds.append(f"set {self._tao_command_} {attr} = {value}")
            else:
                cmds.append(f"set {self._tao_command_} {attr}({index + 1}) = {value}")
        return cmds

    @property
    def set_commands(self) -> list[str]:
        """
        Get all Tao 'set' commands to apply this configuration.

        Returns
        -------
        list of str
        """
        return self.get_set_commands(tao=None)

    def set(
        self,
        tao: Tao,
        *,
        allow_errors: bool = False,
        only_changed: bool = False,
        suppress_plotting: bool = True,
        suppress_lattice_calc: bool = True,
        log: str = "DEBUG",
    ) -> bool:
        """
        Apply this configuration to Tao.

        Parameters
        ----------
        tao : Tao
            The Tao instance to which the configuration will be applied.
        allow_errors : bool, default=False
            Allow individual commands to raise errors.
        only_changed : bool, default=False
            Only apply changes that differ from the current configuration in Tao.
        suppress_plotting : bool, default=True
            Suppress any plotting updates during the commands.
        suppress_lattice_calc : bool, default=True
            Suppress lattice calculations during the commands.
        log : str, default="DEBUG"
            The log level to use during the configuration application.

        Returns
        -------
        success : bool
            Returns True if the configuration was applied without errors.
        """
        cmds = self.get_set_commands(tao=tao if only_changed else None)
        if not cmds:
            return True

        success = True

        tao_global = cast(dict[str, Any], tao.tao_global())
        plot_on = tao_global["plot_on"]
        lat_calc_on = tao_global["lattice_calc_on"]

        if suppress_plotting and plot_on:
            tao.cmd("set global plot_on = F")
        if suppress_lattice_calc and lat_calc_on:
            tao.cmd("set global lattice_calc_on = F")

        log_level: int = getattr(logging, log.upper())

        try:
            for cmd in self.get_set_commands(tao=tao if only_changed else None):
                try:
                    logger.log(log_level, f"Tao> {cmd}")
                    for line in tao.cmd(cmd):
                        logger.log(log_level, f"{line}")
                except Exception as ex:
                    if not allow_errors:
                        raise
                    success = False
                    reason = textwrap.indent(str(ex), "  ")
                    logger.error(f"{cmd!r} failed with:\n{reason}")
        finally:
            if suppress_plotting and plot_on:
                tao.cmd("set global plot_on = T")
            if suppress_lattice_calc and lat_calc_on:
                tao.cmd("set global lattice_calc_on = T")

        return success

    @contextlib.contextmanager
    def set_context(self, tao: Tao):
        """
        Apply this configuration to Tao **only** for the given ``with`` block.

        Examples
        --------

        Set an initial value for a parameter:

        >>> new_state.param = 1
        >>> new_state.set()

        Then temporarily set it to another value, just for the `with` block:

        >>> new_state.param = 3
        >>> with new_state.set_context(tao):
        ...     assert new_state.query(tao).param == 3

        After the ``with`` block, ``param`` will be reset to its previous
        value:

        >>> assert new_state.query(tao).param == 1
        """
        pre_state = self.query(tao)
        for cmd in self.set_commands:
            tao.cmd(cmd)
        yield pre_state
        pre_state.set(tao)


@functools.wraps(pydantic.Field)
def Field(
    attr: str | None = None,
    tao_name: str | None = None,
    **kwargs,
):
    """
    Creates a Pydantic Field based on a pytao parameter.

    Parameters
    ----------
    attr : str or None, optional
        The Python class attribute name.
    tao_name : str or None, optional
        The pytao key associated with the attribute, if it differs from `attr`.
    **kwargs
        Additional keyword arguments passed to the Pydantic Field constructor.

    Returns
    -------
    pydantic.fields.FieldInfo
    """
    if tao_name is not None:
        assert attr is not None
        return pydantic.Field(
            validation_alias=pydantic.AliasChoices(
                attr,
                tao_name,
            ),
            serialization_alias=tao_name,
            **kwargs,
        )
    return pydantic.Field(**kwargs)


@functools.wraps(pydantic.Field)
def ROField(
    attr: str | None = None,
    tao_name: str | None = None,
    **kwargs,
):
    """
    Creates a read-only Pydantic Field based on a pytao parameter.

    Parameters
    ----------
    attr : str or None, optional
        The Python class attribute name.
    tao_name : str or None, optional
        The pytao key associated with the attribute, if it differs from `attr`.
    **kwargs
        Additional keyword arguments passed to the Pydantic Field constructor.

    Returns
    -------
    pydantic.fields.FieldInfo
    """
    return Field(
        attr=attr,
        tao_name=tao_name,
        frozen=True,
        **kwargs,
    )


class Beam(TaoSettableModel):
    """
    Structure which corresponds to Tao `pipe beam`, for example.

    Attributes
    ----------
    always_reinit : bool
    comb_ds_save : float
        Master parameter for %bunch_params_comb(:)%ds_save
    ds_save : float
        Min distance between points.
    dump_at : str
    dump_file : str
    saved_at : str
    track_beam_in_universe : bool
        Beam tracking enabled in this universe?
    track_end : str
    track_start : str
        Tracking start element.
    """

    _tao_command_: ClassVar[str] = "beam"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {"ix_branch": 0, "ix_uni": 1}
    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()

    always_reinit: bool = Field(default=False)
    comb_ds_save: float = Field(
        default=-1.0, description="Master parameter for %bunch_params_comb(:)%ds_save"
    )
    ds_save: float = ROField(default=-1.0, description="Min distance between points.")
    dump_at: str = Field(default="")
    dump_file: str = Field(default="")
    saved_at: str = Field(default="")
    track_beam_in_universe: bool = ROField(
        default=False, description="Beam tracking enabled in this universe?"
    )
    track_end: str = Field(default="")
    track_start: str = Field(default="", description="Tracking start element.")


class BeamInit(TaoSettableModel):
    """
    Structure which corresponds to Tao `pipe beam_init`, for example.

    Attributes
    ----------
    a_emit : float
        a-mode emittance
    a_norm_emit : float
        a-mode normalized emittance (emit * beta * gamma)
    b_emit : float
        b-mode emittance
    b_norm_emit : float
        b-mode normalized emittance (emit * beta * gamma)
    bunch_charge : float
        charge (Coul) in a bunch.
    center : sequence of floats
        Bench phase space center offset relative to reference.
    center_jitter : sequence of floats
        Bunch center rms jitter
    distribution_type : Sequence[str]
        distribution type (in x-px, y-py, and z-pz planes) "ELLIPSE", "KV", "GRID",
        "FILE", "RAN_GAUSS" or "" = "RAN_GAUSS"
    dpz_dz : float
        Correlation of Pz with long position.
    dt_bunch : float
        Time between bunches.
    emit_jitter : sequence of floats
        a and b bunch emittance rms jitter normalized to emittance
    full_6d_coupling_calc : bool
        Use V from 6x6 1-turn mat to match distribution? Else use 4x4 1-turn mat used.
    ix_turn : int
        Turn index used to adjust particles time if needed.
    n_bunch : int
        Number of bunches.
    n_particle : int
        Number of particles per bunch.
    position_file : str
        File with particle positions.
    random_engine : str
        Or 'quasi'. Random number engine to use.
    random_gauss_converter : str
        Or 'quick'. Uniform to gauss conversion method.
    random_sigma_cutoff : float
        Cut-off in sigmas.
    renorm_center : bool
        Renormalize centroid?
    renorm_sigma : bool
        Renormalize sigma?
    sig_pz : float
        pz sigma
    sig_pz_jitter : float
        RMS pz spread jitter
    sig_z : float
        Z sigma in m.
    sig_z_jitter : float
        bunch length RMS jitter
    species : str
        "positron", etc. "" => use referece particle.
    spin : sequence of floats
        Spin (x, y, z)
    t_offset : float
        Time center offset
    use_particle_start : bool
        Use lat%particle_start instead of beam_init%center, %t_offset, and %spin?
    use_t_coords : bool
        If true, the distributions will be taken as in t-coordinates
    use_z_as_t : bool
        Only used if  use_t_coords = .true. If true,  z describes the t distribution If
        false, z describes the s distribution
    """

    _tao_command_: ClassVar[str] = "beam_init"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {"ix_branch": 0, "ix_uni": 1}
    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ("a_emit", "b_emit", "a_norm_emit", "b_norm_emit")

    a_emit: float = Field(default=0.0, description="a-mode emittance")
    a_norm_emit: float = Field(
        default=0.0, description="a-mode normalized emittance (emit * beta * gamma)"
    )
    b_emit: float = Field(default=0.0, description="b-mode emittance")
    b_norm_emit: float = Field(
        default=0.0, description="b-mode normalized emittance (emit * beta * gamma)"
    )
    bunch_charge: float = Field(default=0.0, description="charge (Coul) in a bunch.")
    center: FloatSequence = Field(
        default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        max_length=6,
        description="Bench phase space center offset relative to reference.",
    )
    center_jitter: FloatSequence = Field(
        default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], max_length=6, description="Bunch center rms jitter"
    )
    distribution_type: Sequence[str] = ROField(
        default=["RAN_GAUSS", "RAN_GAUSS", "RAN_GAUSS"],
        max_length=3,
        description=(
            "distribution type (in x-px, y-py, and z-pz planes) 'ELLIPSE', 'KV', "
            "'GRID', 'FILE', 'RAN_GAUSS' or '' = 'RAN_GAUSS'"
        ),
    )
    dpz_dz: float = Field(default=0.0, description="Correlation of Pz with long position.")
    dt_bunch: float = Field(default=0.0, description="Time between bunches.")
    emit_jitter: FloatSequence = Field(
        default=[0.0, 0.0],
        max_length=2,
        description="a and b bunch emittance rms jitter normalized to emittance",
    )
    full_6d_coupling_calc: bool = Field(
        default=False,
        description="Use V from 6x6 1-turn mat to match distribution? Else use 4x4 1-turn mat used.",
    )
    ix_turn: int = Field(
        default=0, description="Turn index used to adjust particles time if needed."
    )
    n_bunch: int = Field(default=0, description="Number of bunches.")
    n_particle: int = Field(default=0, description="Number of particles per bunch.")
    position_file: str = Field(default="", description="File with particle positions.")
    random_engine: str = Field(
        default="pseudo", description="Or 'quasi'. Random number engine to use."
    )
    random_gauss_converter: str = Field(
        default="exact", description="Or 'quick'. Uniform to gauss conversion method."
    )
    random_sigma_cutoff: float = Field(default=-1.0, description="Cut-off in sigmas.")
    renorm_center: bool = Field(default=True, description="Renormalize centroid?")
    renorm_sigma: bool = Field(default=True, description="Renormalize sigma?")
    sig_pz: float = Field(default=0.0, description="pz sigma")
    sig_pz_jitter: float = Field(default=0.0, description="RMS pz spread jitter")
    sig_z: float = Field(default=0.0, description="Z sigma in m.")
    sig_z_jitter: float = Field(default=0.0, description="bunch length RMS jitter")
    species: str = Field(default="", description="'positron', etc. '' => use referece particle.")
    spin: FloatSequence = Field(default=[0.0, 0.0, 0.0], max_length=3, description="Spin (x, y, z)")
    t_offset: float = Field(default=0.0, description="Time center offset")
    use_particle_start: bool = Field(
        default=False,
        description="Use lat%particle_start instead of beam_init%center, %t_offset, and %spin?",
    )
    use_t_coords: bool = Field(
        default=False, description="If true, the distributions will be taken as in t-coordinates"
    )
    use_z_as_t: bool = Field(
        default=False,
        description=(
            "Only used if  use_t_coords = .true. If true,  z describes the t "
            "distribution If false, z describes the s distribution"
        ),
    )


class BmadCom(TaoSettableModel):
    """
    Structure which corresponds to Tao `pipe bmad_com`, for example.

    Attributes
    ----------
    abs_tol_adaptive_tracking : float
        Runge-Kutta tracking absolute tolerance.
    abs_tol_tracking : float
        Closed orbit absolute tolerance.
    absolute_time_tracking : bool
        Absolute or relative time tracking?
    aperture_limit_on : bool
        Use apertures in tracking?
    auto_bookkeeper : bool
        Automatic bookkeeping?
    autoscale_amp_abs_tol : float
        Autoscale absolute amplitude tolerance (eV).
    autoscale_amp_rel_tol : float
        Autoscale relative amplitude tolerance
    autoscale_phase_tol : float
        Autoscale phase tolerance.
    conserve_taylor_maps : bool
        Enable bookkeeper to set ele%taylor_map_includes_offsets = F?
    convert_to_kinetic_momentum : bool
        Cancel kicks due to finite vector potential when doing symplectic tracking? Set
        to True to test symp_lie_bmad against runge_kutta.
    csr_and_space_charge_on : bool
        Space charge switch.
    d_orb : sequence of floats
        Orbit deltas for the mat6 via tracking calc.
    debug : bool
        Used for code debugging.
    default_ds_step : float
        Default integration step for eles without an explicit step calc.
    default_integ_order : int
        PTC integration order.
    electric_dipole_moment : float
        Particle's EDM. Call set_ptc to transfer value to PTC.
    fatal_ds_adaptive_tracking : float
        If actual step size is below this particle is lost.
    init_ds_adaptive_tracking : float
        Initial step size
    lr_wakes_on : bool
        Long range wakefields
    max_aperture_limit : float
        Max Aperture.
    max_num_runge_kutta_step : int
        Maximum number of RK steps before particle is considered lost.
    min_ds_adaptive_tracking : float
        Min step size to take.
    radiation_damping_on : bool
        Radiation damping toggle.
    radiation_fluctuations_on : bool
        Radiation fluctuations toggle.
    rel_tol_adaptive_tracking : float
        Runge-Kutta tracking relative tolerance.
    rel_tol_tracking : float
        Closed orbit relative tolerance.
    rf_phase_below_transition_ref : bool
        Autoscale uses below transition stable point for RFCavities?
    runge_kutta_order : int
        Runge Kutta order.
    sad_amp_max : float
        Used in sad_mult step length calc.
    sad_eps_scale : float
        Used in sad_mult step length calc.
    sad_n_div_max : int
        Used in sad_mult step length calc.
    significant_length : float
        meter
    spin_sokolov_ternov_flipping_on : bool
        Spin flipping during synchrotron radiation emission?
    spin_tracking_on : bool
        spin tracking?
    sr_wakes_on : bool
        Short range wakefields?
    synch_rad_scale : float
        Synch radiation kick scale. 1 => normal, 0 => no kicks.
    taylor_order : int
        Taylor order to use. 0 -> default = ptc_private%taylor_order_saved.
    """

    _tao_command_: ClassVar[str] = "bmad_com"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()

    abs_tol_adaptive_tracking: float = Field(
        default=1e-10, description="Runge-Kutta tracking absolute tolerance."
    )
    abs_tol_tracking: float = Field(default=1e-12, description="Closed orbit absolute tolerance.")
    absolute_time_tracking: bool = Field(
        default=False, description="Absolute or relative time tracking?"
    )
    aperture_limit_on: bool = Field(default=True, description="Use apertures in tracking?")
    auto_bookkeeper: bool = Field(default=True, description="Automatic bookkeeping?")
    autoscale_amp_abs_tol: float = Field(
        default=0.1, description="Autoscale absolute amplitude tolerance (eV)."
    )
    autoscale_amp_rel_tol: float = Field(
        default=1e-06, description="Autoscale relative amplitude tolerance"
    )
    autoscale_phase_tol: float = Field(default=1e-05, description="Autoscale phase tolerance.")
    conserve_taylor_maps: bool = Field(
        default=True, description="Enable bookkeeper to set ele%taylor_map_includes_offsets = F?"
    )
    convert_to_kinetic_momentum: bool = Field(
        default=False,
        description=(
            "Cancel kicks due to finite vector potential when doing symplectic "
            "tracking? Set to True to test symp_lie_bmad against runge_kutta."
        ),
    )
    csr_and_space_charge_on: bool = Field(default=False, description="Space charge switch.")
    d_orb: FloatSequence = Field(
        default=[1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05],
        max_length=6,
        description="Orbit deltas for the mat6 via tracking calc.",
    )
    debug: bool = Field(default=False, description="Used for code debugging.")
    default_ds_step: float = Field(
        default=0.2, description="Default integration step for eles without an explicit step calc."
    )
    default_integ_order: int = Field(default=2, description="PTC integration order.")
    electric_dipole_moment: float = Field(
        default=0.0, description="Particle's EDM. Call set_ptc to transfer value to PTC."
    )
    fatal_ds_adaptive_tracking: float = Field(
        default=1e-08, description="If actual step size is below this particle is lost."
    )
    init_ds_adaptive_tracking: float = Field(default=0.001, description="Initial step size")
    lr_wakes_on: bool = Field(default=True, description="Long range wakefields")
    max_aperture_limit: float = Field(default=1000.0, description="Max Aperture.")
    max_num_runge_kutta_step: int = Field(
        default=10000, description="Maximum number of RK steps before particle is considered lost."
    )
    min_ds_adaptive_tracking: float = Field(default=0.0, description="Min step size to take.")
    radiation_damping_on: bool = Field(default=False, description="Radiation damping toggle.")
    radiation_fluctuations_on: bool = Field(
        default=False, description="Radiation fluctuations toggle."
    )
    rel_tol_adaptive_tracking: float = Field(
        default=1e-08, description="Runge-Kutta tracking relative tolerance."
    )
    rel_tol_tracking: float = Field(default=1e-09, description="Closed orbit relative tolerance.")
    rf_phase_below_transition_ref: bool = Field(
        default=False, description="Autoscale uses below transition stable point for RFCavities?"
    )
    runge_kutta_order: int = Field(default=4, description="Runge Kutta order.")
    sad_amp_max: float = Field(default=0.05, description="Used in sad_mult step length calc.")
    sad_eps_scale: float = Field(default=0.005, description="Used in sad_mult step length calc.")
    sad_n_div_max: int = Field(default=1000, description="Used in sad_mult step length calc.")
    significant_length: float = Field(default=1e-10, description="meter")
    spin_sokolov_ternov_flipping_on: bool = Field(
        default=False, description="Spin flipping during synchrotron radiation emission?"
    )
    spin_tracking_on: bool = Field(default=False, description="spin tracking?")
    sr_wakes_on: bool = Field(default=True, description="Short range wakefields?")
    synch_rad_scale: float = Field(
        default=1.0, description="Synch radiation kick scale. 1 => normal, 0 => no kicks."
    )
    taylor_order: int = Field(
        default=0, description="Taylor order to use. 0 -> default = ptc_private%taylor_order_saved."
    )


class ElementBunchParams(TaoModel):
    """
    Structure which corresponds to Tao `pipe bunch_params 1`, for example.

    Attributes
    ----------
    beam_saved : bool
    centroid_beta : float
    centroid_p0c : float
    centroid_t : float
    centroid_vec_1 : float
    centroid_vec_2 : float
    centroid_vec_3 : float
    centroid_vec_4 : float
    centroid_vec_5 : float
    centroid_vec_6 : float
    charge_live : float
        Charge of all non-lost particle
    direction : int
    ix_ele : int
        Lattice element where params evaluated at.
    location : str
        Location in element: upstream_end$, inside$, or downstream_end$
    n_particle_live : int
        Number of non-lost particles
    n_particle_lost_in_ele : int
        Number lost in element (not calculated by Bmad)
    n_particle_tot : int
        Total number of particles
    rel_max_1 : float
    rel_max_2 : float
    rel_max_3 : float
    rel_max_4 : float
    rel_max_5 : float
    rel_max_6 : float
    rel_min_1 : float
    rel_min_2 : float
    rel_min_3 : float
    rel_min_4 : float
    rel_min_5 : float
    rel_min_6 : float
    s : float
        Longitudinal position.
    sigma_11 : float
    sigma_12 : float
    sigma_13 : float
    sigma_14 : float
    sigma_15 : float
    sigma_16 : float
    sigma_21 : float
    sigma_22 : float
    sigma_23 : float
    sigma_24 : float
    sigma_25 : float
    sigma_26 : float
    sigma_31 : float
    sigma_32 : float
    sigma_33 : float
    sigma_34 : float
    sigma_35 : float
    sigma_36 : float
    sigma_41 : float
    sigma_42 : float
    sigma_43 : float
    sigma_44 : float
    sigma_45 : float
    sigma_46 : float
    sigma_51 : float
    sigma_52 : float
    sigma_53 : float
    sigma_54 : float
    sigma_55 : float
    sigma_56 : float
    sigma_61 : float
    sigma_62 : float
    sigma_63 : float
    sigma_64 : float
    sigma_65 : float
    sigma_66 : float
    sigma_t : float
        RMS of time spread.
    species : str
    t : float
        Time.
    twiss_alpha_a : float
    twiss_alpha_b : float
    twiss_alpha_c : float
    twiss_alpha_x : float
    twiss_alpha_y : float
    twiss_alpha_z : float
    twiss_beta_a : float
    twiss_beta_b : float
    twiss_beta_c : float
    twiss_beta_x : float
    twiss_beta_y : float
    twiss_beta_z : float
    twiss_emit_a : float
    twiss_emit_b : float
    twiss_emit_c : float
    twiss_emit_x : float
    twiss_emit_y : float
    twiss_emit_z : float
    twiss_eta_a : float
    twiss_eta_b : float
    twiss_eta_c : float
    twiss_eta_x : float
    twiss_eta_y : float
    twiss_eta_z : float
    twiss_etap_a : float
    twiss_etap_b : float
    twiss_etap_c : float
    twiss_etap_x : float
    twiss_etap_y : float
    twiss_etap_z : float
    twiss_gamma_a : float
    twiss_gamma_b : float
    twiss_gamma_c : float
    twiss_gamma_x : float
    twiss_gamma_y : float
    twiss_gamma_z : float
    twiss_norm_emit_a : float
    twiss_norm_emit_b : float
    twiss_norm_emit_c : float
    twiss_norm_emit_x : float
    twiss_norm_emit_y : float
    twiss_norm_emit_z : float
    twiss_phi_a : float
    twiss_phi_b : float
    twiss_phi_c : float
    twiss_phi_x : float
    twiss_phi_y : float
    twiss_phi_z : float
    twiss_sigma_a : float
    twiss_sigma_b : float
    twiss_sigma_c : float
    twiss_sigma_p_a : float
    twiss_sigma_p_b : float
    twiss_sigma_p_c : float
    twiss_sigma_p_x : float
    twiss_sigma_p_y : float
    twiss_sigma_p_z : float
    twiss_sigma_x : float
    twiss_sigma_y : float
    twiss_sigma_z : float
    """

    _tao_command_: ClassVar[str] = "bunch_params"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}

    beam_saved: bool = Field(default=False)
    centroid_beta: float = ROField(default=0.0)
    centroid_p0c: float = ROField(default=0.0)
    centroid_t: float = ROField(default=0.0)
    centroid_vec_1: float = ROField(default=0.0)
    centroid_vec_2: float = ROField(default=0.0)
    centroid_vec_3: float = ROField(default=0.0)
    centroid_vec_4: float = ROField(default=0.0)
    centroid_vec_5: float = ROField(default=0.0)
    centroid_vec_6: float = ROField(default=0.0)
    charge_live: float = ROField(default=0.0, description="Charge of all non-lost particle")
    direction: int = ROField(default=0)
    ix_ele: int = ROField(default=-1, description="Lattice element where params evaluated at.")
    location: str = ROField(
        default="", description="Location in element: upstream_end$, inside$, or downstream_end$"
    )
    n_particle_live: int = ROField(default=0, description="Number of non-lost particles")
    n_particle_lost_in_ele: int = ROField(
        default=0, description="Number lost in element (not calculated by Bmad)"
    )
    n_particle_tot: int = ROField(default=0, description="Total number of particles")
    rel_max_1: float = ROField(default=0.0)
    rel_max_2: float = ROField(default=0.0)
    rel_max_3: float = ROField(default=0.0)
    rel_max_4: float = ROField(default=0.0)
    rel_max_5: float = ROField(default=0.0)
    rel_max_6: float = ROField(default=0.0)
    rel_min_1: float = ROField(default=0.0)
    rel_min_2: float = ROField(default=0.0)
    rel_min_3: float = ROField(default=0.0)
    rel_min_4: float = ROField(default=0.0)
    rel_min_5: float = ROField(default=0.0)
    rel_min_6: float = ROField(default=0.0)
    s: float = ROField(default=-1.0, description="Longitudinal position.")
    sigma_11: float = ROField(default=0.0)
    sigma_12: float = ROField(default=0.0)
    sigma_13: float = ROField(default=0.0)
    sigma_14: float = ROField(default=0.0)
    sigma_15: float = ROField(default=0.0)
    sigma_16: float = ROField(default=0.0)
    sigma_21: float = ROField(default=0.0)
    sigma_22: float = ROField(default=0.0)
    sigma_23: float = ROField(default=0.0)
    sigma_24: float = ROField(default=0.0)
    sigma_25: float = ROField(default=0.0)
    sigma_26: float = ROField(default=0.0)
    sigma_31: float = ROField(default=0.0)
    sigma_32: float = ROField(default=0.0)
    sigma_33: float = ROField(default=0.0)
    sigma_34: float = ROField(default=0.0)
    sigma_35: float = ROField(default=0.0)
    sigma_36: float = ROField(default=0.0)
    sigma_41: float = ROField(default=0.0)
    sigma_42: float = ROField(default=0.0)
    sigma_43: float = ROField(default=0.0)
    sigma_44: float = ROField(default=0.0)
    sigma_45: float = ROField(default=0.0)
    sigma_46: float = ROField(default=0.0)
    sigma_51: float = ROField(default=0.0)
    sigma_52: float = ROField(default=0.0)
    sigma_53: float = ROField(default=0.0)
    sigma_54: float = ROField(default=0.0)
    sigma_55: float = ROField(default=0.0)
    sigma_56: float = ROField(default=0.0)
    sigma_61: float = ROField(default=0.0)
    sigma_62: float = ROField(default=0.0)
    sigma_63: float = ROField(default=0.0)
    sigma_64: float = ROField(default=0.0)
    sigma_65: float = ROField(default=0.0)
    sigma_66: float = ROField(default=0.0)
    sigma_t: float = ROField(default=0.0, description="RMS of time spread.")
    species: str = ROField(default="")
    t: float = ROField(default=-1.0, description="Time.")
    twiss_alpha_a: float = ROField(default=0.0)
    twiss_alpha_b: float = ROField(default=0.0)
    twiss_alpha_c: float = ROField(default=0.0)
    twiss_alpha_x: float = ROField(default=0.0)
    twiss_alpha_y: float = ROField(default=0.0)
    twiss_alpha_z: float = ROField(default=0.0)
    twiss_beta_a: float = ROField(default=0.0)
    twiss_beta_b: float = ROField(default=0.0)
    twiss_beta_c: float = ROField(default=0.0)
    twiss_beta_x: float = ROField(default=0.0)
    twiss_beta_y: float = ROField(default=0.0)
    twiss_beta_z: float = ROField(default=0.0)
    twiss_emit_a: float = ROField(default=0.0)
    twiss_emit_b: float = ROField(default=0.0)
    twiss_emit_c: float = ROField(default=0.0)
    twiss_emit_x: float = ROField(default=0.0)
    twiss_emit_y: float = ROField(default=0.0)
    twiss_emit_z: float = ROField(default=0.0)
    twiss_eta_a: float = ROField(default=0.0)
    twiss_eta_b: float = ROField(default=0.0)
    twiss_eta_c: float = ROField(default=0.0)
    twiss_eta_x: float = ROField(default=0.0)
    twiss_eta_y: float = ROField(default=0.0)
    twiss_eta_z: float = ROField(default=0.0)
    twiss_etap_a: float = ROField(default=0.0)
    twiss_etap_b: float = ROField(default=0.0)
    twiss_etap_c: float = ROField(default=0.0)
    twiss_etap_x: float = ROField(default=0.0)
    twiss_etap_y: float = ROField(default=0.0)
    twiss_etap_z: float = ROField(default=0.0)
    twiss_gamma_a: float = ROField(default=0.0)
    twiss_gamma_b: float = ROField(default=0.0)
    twiss_gamma_c: float = ROField(default=0.0)
    twiss_gamma_x: float = ROField(default=0.0)
    twiss_gamma_y: float = ROField(default=0.0)
    twiss_gamma_z: float = ROField(default=0.0)
    twiss_norm_emit_a: float = ROField(default=0.0)
    twiss_norm_emit_b: float = ROField(default=0.0)
    twiss_norm_emit_c: float = ROField(default=0.0)
    twiss_norm_emit_x: float = ROField(default=0.0)
    twiss_norm_emit_y: float = ROField(default=0.0)
    twiss_norm_emit_z: float = ROField(default=0.0)
    twiss_phi_a: float = ROField(default=0.0)
    twiss_phi_b: float = ROField(default=0.0)
    twiss_phi_c: float = ROField(default=0.0)
    twiss_phi_x: float = ROField(default=0.0)
    twiss_phi_y: float = ROField(default=0.0)
    twiss_phi_z: float = ROField(default=0.0)
    twiss_sigma_a: float = ROField(default=0.0)
    twiss_sigma_b: float = ROField(default=0.0)
    twiss_sigma_c: float = ROField(default=0.0)
    twiss_sigma_p_a: float = ROField(default=0.0)
    twiss_sigma_p_b: float = ROField(default=0.0)
    twiss_sigma_p_c: float = ROField(default=0.0)
    twiss_sigma_p_x: float = ROField(default=0.0)
    twiss_sigma_p_y: float = ROField(default=0.0)
    twiss_sigma_p_z: float = ROField(default=0.0)
    twiss_sigma_x: float = ROField(default=0.0)
    twiss_sigma_y: float = ROField(default=0.0)
    twiss_sigma_z: float = ROField(default=0.0)


class ElementChamberWall(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:chamber_wall 1 1 x`, for example.

    Attributes
    ----------
    longitudinal_position : float
    section : int
    z1 : float
    z2_neg : float
    """

    _tao_command_: ClassVar[str] = "ele_chamber_wall"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}

    longitudinal_position: float = ROField(default=0.0)
    section: int = ROField(default=0)
    z1: float = ROField(default=0.0)
    z2_neg: float = ROField(default=0.0, attr="z2_neg", tao_name="-z2")


class ElementGridField(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:grid_field G1 1 base`, for example.

    Attributes
    ----------
    curved_ref_frame : bool
    dr : sequence of floats
        Grid spacing.
    ele_anchor_pt : str
        anchor_beginning$, anchor_center$, or anchor_end$
    field_scale : float
        Factor to scale the fields by
    field_type : str
        or magnetic$ or electric$
    file : str
    grid_field_geometry : str
    harmonic : int
        Harmonic of fundamental for AC fields.
    interpolation_order : int
        Possibilities are 1 or 3.
    master_parameter : str
        Master parameter in ele%value(:) array to use for scaling the field.
    phi0_fieldmap : float
        Mode oscillates as: twopi * (f * t + phi0_fieldmap)
    r0 : sequence of floats
        Field origin relative to ele_anchor_pt.
    """

    _tao_command_: ClassVar[str] = "ele_grid_field"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}

    curved_ref_frame: bool = Field(default=False)
    dr: FloatSequence = Field(default=[0.0, 0.0, 0.0], max_length=3, description="Grid spacing.")
    ele_anchor_pt: str = Field(
        default="", description="anchor_beginning$, anchor_center$, or anchor_end$"
    )
    field_scale: float = Field(default=1.0, description="Factor to scale the fields by")
    field_type: str = Field(default="", description="or magnetic$ or electric$")
    file: str = Field(default="")
    grid_field_geometry: str = ROField(
        default="", attr="grid_field_geometry", tao_name="grid_field^geometry"
    )
    harmonic: int = Field(default=0, description="Harmonic of fundamental for AC fields.")
    interpolation_order: int = Field(default=1, description="Possibilities are 1 or 3.")
    master_parameter: str = Field(
        default=0,
        description="Master parameter in ele%value(:) array to use for scaling the field.",
    )
    phi0_fieldmap: float = Field(
        default=0.0, description="Mode oscillates as: twopi * (f * t + phi0_fieldmap)"
    )
    r0: FloatSequence = Field(
        default=[0.0, 0.0, 0.0], max_length=3, description="Field origin relative to ele_anchor_pt."
    )


class ElementGridFieldPoints(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:grid_field G1 1 points`, for example.

    Attributes
    ----------
    data : FloatSequence
    i : int
    j : int
    k : int
    """

    _tao_command_: ClassVar[str] = "ele_grid_field"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}

    data: FloatSequence = ROField(default_factory=list)
    i: int = ROField(default=0)
    j: int = ROField(default=0)
    k: int = ROField(default=0)


class ElementHead(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:head 1`, for example.

    Attributes
    ----------
    alias : str
        Another name.
    descrip : str
        Description string.
    has_ab_multipoles : bool
    has_ac_kick : bool
    has_control : bool
    has_floor : bool
    has_kt_multipoles : bool
    has_lord_slave : bool
    has_mat6 : bool
    has_methods : bool
    has_multipoles_elec : bool
    has_photon : bool
    has_spin_taylor : bool
    has_taylor : bool
    has_twiss : bool
    has_wake : bool
    has_wall3d : int
    is_on : bool
        For turning element on/off.
    ix_branch : int
        Index in lat%branch(:) array. Note: lat%ele => lat%branch(0).
    ix_ele : int
        Index in branch ele(0:) array. Set to ix_slice_slave$ = -2 for slice_slave$
        elements.
    key : str
        Element class (quadrupole, etc.).
    name : str
        name of element.
    num_cartesian_map : int
    num_cylindrical_map : int
    num_gen_grad_map : int
    num_grid_field : int
    ref_time : float
        Time ref particle passes exit end.
    s : float
        longitudinal ref position at the exit end.
    s_start : float
        longitudinal ref position at entrance_end
    type : str
        type name.
    universe : int
    """

    _tao_command_: ClassVar[str] = "ele_head"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}

    alias: str = Field(default="", description="Another name.")
    descrip: str = Field(default="", description="Description string.")
    has_ab_multipoles: bool = ROField(
        default=False, attr="has_ab_multipoles", tao_name="has#ab_multipoles"
    )
    has_ac_kick: bool = ROField(default=False, attr="has_ac_kick", tao_name="has#ac_kick")
    has_control: bool = ROField(default=False, attr="has_control", tao_name="has#control")
    has_floor: bool = ROField(default=False, attr="has_floor", tao_name="has#floor")
    has_kt_multipoles: bool = ROField(
        default=False, attr="has_kt_multipoles", tao_name="has#kt_multipoles"
    )
    has_lord_slave: bool = ROField(default=False, attr="has_lord_slave", tao_name="has#lord_slave")
    has_mat6: bool = ROField(default=False, attr="has_mat6", tao_name="has#mat6")
    has_methods: bool = ROField(default=False, attr="has_methods", tao_name="has#methods")
    has_multipoles_elec: bool = ROField(
        default=False, attr="has_multipoles_elec", tao_name="has#multipoles_elec"
    )
    has_photon: bool = ROField(default=False, attr="has_photon", tao_name="has#photon")
    has_spin_taylor: bool = ROField(
        default=False, attr="has_spin_taylor", tao_name="has#spin_taylor"
    )
    has_taylor: bool = ROField(default=False, attr="has_taylor", tao_name="has#taylor")
    has_twiss: bool = ROField(default=False, attr="has_twiss", tao_name="has#twiss")
    has_wake: bool = ROField(default=False, attr="has_wake", tao_name="has#wake")
    has_wall3d: int = ROField(default=0, attr="has_wall3d", tao_name="has#wall3d")
    is_on: bool = Field(default=True, description="For turning element on/off.")
    ix_branch: int = ROField(
        default=0,
        description="Index in lat%branch(:) array. Note: lat%ele => lat%branch(0).",
        attr="ix_branch",
        tao_name="1^ix_branch",
    )
    ix_ele: int = ROField(
        default=-1,
        description=(
            "Index in branch ele(0:) array. Set to ix_slice_slave$ = -2 for slice_slave$ elements."
        ),
    )
    key: str = ROField(default=0, description="Element class (quadrupole, etc.).")
    name: str = ROField(default="<Initialized>", description="name of element.")
    num_cartesian_map: int = ROField(
        default=0, attr="num_cartesian_map", tao_name="num#cartesian_map"
    )
    num_cylindrical_map: int = ROField(
        default=0, attr="num_cylindrical_map", tao_name="num#cylindrical_map"
    )
    num_gen_grad_map: int = ROField(default=0, attr="num_gen_grad_map", tao_name="num#gen_grad_map")
    num_grid_field: int = ROField(default=0, attr="num_grid_field", tao_name="num#grid_field")
    ref_time: float = ROField(default=0.0, description="Time ref particle passes exit end.")
    s: float = ROField(default=0.0, description="longitudinal ref position at the exit end.")
    s_start: float = ROField(default=0.0, description="longitudinal ref position at entrance_end")
    type: str = Field(default="", description="type name.")
    universe: int = ROField(default=0)


class ElementLordSlave(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:lord_slave 1 1 x`, for example.

    Attributes
    ----------
    key : str
    location_name : str
    name : str
    status : str
    type : str
    """

    _tao_command_: ClassVar[str] = "ele_lord_slave"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}

    key: str = ROField(default="")
    location_name: str = ROField(default="")
    name: str = ROField(default="")
    status: str = ROField(default="")
    type: str = ROField(default="")


class ElementMat6(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:mat6 1 mat6`, for example.

    Attributes
    ----------
    data_1 : sequence of floats
    data_2 : sequence of floats
    data_3 : sequence of floats
    data_4 : sequence of floats
    data_5 : sequence of floats
    data_6 : sequence of floats
    """

    _tao_command_: ClassVar[str] = "ele_mat6"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}

    data_1: FloatSequence = ROField(default_factory=list, max_length=6, attr="data_1", tao_name="1")
    data_2: FloatSequence = ROField(default_factory=list, max_length=6, attr="data_2", tao_name="2")
    data_3: FloatSequence = ROField(default_factory=list, max_length=6, attr="data_3", tao_name="3")
    data_4: FloatSequence = ROField(default_factory=list, max_length=6, attr="data_4", tao_name="4")
    data_5: FloatSequence = ROField(default_factory=list, max_length=6, attr="data_5", tao_name="5")
    data_6: FloatSequence = ROField(default_factory=list, max_length=6, attr="data_6", tao_name="6")


class ElementMat6Error(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:mat6 1 err`, for example.

    Attributes
    ----------
    symplectic_error : float
    """

    _tao_command_: ClassVar[str] = "ele_mat6"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}

    symplectic_error: float = ROField(default=0.0)


class ElementMat6Vec0(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:mat6 1 vec0`, for example.

    Attributes
    ----------
    vec0 : sequence of floats
    """

    _tao_command_: ClassVar[str] = "ele_mat6"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}

    vec0: FloatSequence = ROField(default_factory=list, max_length=6)


class ElementMultipoles_Data(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:multipoles 13`, for example.

    Attributes
    ----------
    an_equiv : float
    bn_equiv : float
    index : int
    knl : float
    knl_w_tilt : float
    tn : float
    tn_w_tilt : float
    """

    _tao_command_: ClassVar[str] = "ele_multipoles"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}

    an_equiv: float = ROField(default=0.0, attr="an_equiv", tao_name="An (equiv)")
    bn_equiv: float = ROField(default=0.0, attr="bn_equiv", tao_name="Bn (equiv)")
    index: int = ROField(default=0)
    knl: float = ROField(default=0.0, attr="knl", tao_name="KnL")
    knl_w_tilt: float = ROField(default=0.0, attr="knl_w_tilt", tao_name="KnL (w/Tilt)")
    tn: float = ROField(default=0.0, attr="tn", tao_name="Tn")
    tn_w_tilt: float = ROField(default=0.0, attr="tn_w_tilt", tao_name="Tn (w/Tilt)")


class ElementMultipoles(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:multipoles 13`, for example.

    Attributes
    ----------
    data : ElementMultipoles_Data
        Structure which corresponds to Tao `pipe ele:multipoles 13`, for example.
    multipoles_on : bool
        For turning multipoles on/off
    scale_multipoles : bool or None
        Are ab_multipoles within other elements (EG: quads, etc.) scaled by the
        strength of the element?
    """

    _tao_command_: ClassVar[str] = "ele_multipoles"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}

    data: Sequence[ElementMultipoles_Data] = ROField(
        default_factory=list,
        description="Structure which corresponds to Tao `pipe ele:multipoles 13`, for example.",
    )
    multipoles_on: bool = Field(default=True, description="For turning multipoles on/off")
    scale_multipoles: bool | None = Field(
        default=None,
        description=(
            "Are ab_multipoles within other elements (EG: quads, etc.) scaled by the "
            "strength of the element?"
        ),
    )


class ElementMultipolesAB_Data(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:multipoles 4`, for example.

    Attributes
    ----------
    an : float
    an_w_tilt : float
    bn : float
    bn_w_tilt : float
    index : int
    knl_equiv : float
    tn_equiv : float
    """

    _tao_command_: ClassVar[str] = "ele_multipoles"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}

    an: float = ROField(default=0.0, attr="an", tao_name="An")
    an_w_tilt: float = ROField(default=0.0, attr="an_w_tilt", tao_name="An (w/Tilt)")
    bn: float = ROField(default=0.0, attr="bn", tao_name="Bn")
    bn_w_tilt: float = ROField(default=0.0, attr="bn_w_tilt", tao_name="Bn (w/Tilt)")
    index: int = ROField(default=0)
    knl_equiv: float = ROField(default=0.0, attr="knl_equiv", tao_name="KnL (equiv)")
    tn_equiv: float = ROField(default=0.0, attr="tn_equiv", tao_name="Tn (equiv)")


class ElementMultipolesAB(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:multipoles 4`, for example.

    Attributes
    ----------
    data : ElementMultipolesAB_Data
        Structure which corresponds to Tao `pipe ele:multipoles 4`, for example.
    multipoles_on : bool
        For turning multipoles on/off
    """

    _tao_command_: ClassVar[str] = "ele_multipoles"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}

    data: Sequence[ElementMultipolesAB_Data] = ROField(
        default_factory=list,
        description="Structure which corresponds to Tao `pipe ele:multipoles 4`, for example.",
    )
    multipoles_on: bool = Field(default=True, description="For turning multipoles on/off")


class ElementMultipolesScaled_Data(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:multipoles 16`, for example.

    Attributes
    ----------
    an : float
    an_scaled : float
    an_w_tilt : float
    bn : float
    bn_scaled : float
    bn_w_tilt : float
    index : int
    knl_equiv : float
    tn_equiv : float
    """

    _tao_command_: ClassVar[str] = "ele_multipoles"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}

    an: float = ROField(default=0.0, attr="an", tao_name="An")
    an_scaled: float = ROField(default=0.0, attr="an_scaled", tao_name="An (Scaled)")
    an_w_tilt: float = ROField(default=0.0, attr="an_w_tilt", tao_name="An (w/Tilt)")
    bn: float = ROField(default=0.0, attr="bn", tao_name="Bn")
    bn_scaled: float = ROField(default=0.0, attr="bn_scaled", tao_name="Bn (Scaled)")
    bn_w_tilt: float = ROField(default=0.0, attr="bn_w_tilt", tao_name="Bn (w/Tilt)")
    index: int = ROField(default=0)
    knl_equiv: float = ROField(default=0.0, attr="knl_equiv", tao_name="KnL (equiv)")
    tn_equiv: float = ROField(default=0.0, attr="tn_equiv", tao_name="Tn (equiv)")


class ElementMultipolesScaled(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:multipoles 16`, for example.

    Attributes
    ----------
    data : ElementMultipolesScaled_Data
        Structure which corresponds to Tao `pipe ele:multipoles 16`, for example.
    multipoles_on : bool
        For turning multipoles on/off
    scale_multipoles : bool
        Are ab_multipoles within other elements (EG: quads, etc.) scaled by the
        strength of the element?
    """

    _tao_command_: ClassVar[str] = "ele_multipoles"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}

    data: Sequence[ElementMultipolesScaled_Data] = ROField(
        default_factory=list,
        description="Structure which corresponds to Tao `pipe ele:multipoles 16`, for example.",
    )
    multipoles_on: bool = Field(default=True, description="For turning multipoles on/off")
    scale_multipoles: bool = Field(
        default=True,
        description=(
            "Are ab_multipoles within other elements (EG: quads, etc.) scaled by the "
            "strength of the element?"
        ),
    )


class ElementOrbit(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:orbit 1`, for example.

    Attributes
    ----------
    beta : float
        Velocity / c_light.
    charge : float
        Macroparticle weight (which is different from particle species charge). For
        some space charge calcs the weight is in Coulombs.
    direction : int
        +1 or -1. Sign of longitudinal direction of motion (ds/dt). This is independent
        of the element orientation.
    dt_ref : float
        Used in: * time tracking for computing z. * by coherent photons =
        path_length/c_light.
    field : sequence of floats
        Photon E-field intensity (x,y).
    ix_ele : int
        Index of the lattice element the particle is in. May be -1 if element is not
        associated with a lattice.
    location : str
        upstream_end$, inside$, or downstream_end$
    p0c : float
        For non-photons: Reference momentum. For photons: Photon momentum (not
        reference).
    phase : sequence of floats
        Photon E-field phase (x,y).
    px : float
    py : float
    pz : float
    s : float
        Longitudinal position
    species : str
        positron$, proton$, etc.
    spin : sequence of floats
        Spin.
    state : str
        alive$, lost$, lost_neg_x_aperture$, lost_pz$, etc.
    t : float
        Absolute time (not relative to reference). Note: Quad precision!
    x : float
    y : float
    z : float
    """

    _tao_command_: ClassVar[str] = "ele_orbit"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}

    beta: float = ROField(default=-1.0, description="Velocity / c_light.")
    charge: float = ROField(
        default=0.0,
        description=(
            "Macroparticle weight (which is different from particle species charge). "
            "For some space charge calcs the weight is in Coulombs."
        ),
    )
    direction: int = ROField(
        default=1,
        description=(
            "+1 or -1. Sign of longitudinal direction of motion (ds/dt). This is "
            "independent of the element orientation."
        ),
    )
    dt_ref: float = ROField(
        default=0.0,
        description=(
            "Used in: * time tracking for computing z. * by coherent photons = path_length/c_light."
        ),
    )
    field: FloatSequence = ROField(
        default=[0.0, 0.0], max_length=2, description="Photon E-field intensity (x,y)."
    )
    ix_ele: int = ROField(
        default=-1,
        description=(
            "Index of the lattice element the particle is in. May be -1 if element is "
            "not associated with a lattice."
        ),
    )
    location: str = ROField(default="", description="upstream_end$, inside$, or downstream_end$")
    p0c: float = ROField(
        default=0.0,
        description=(
            "For non-photons: Reference momentum. For photons: Photon momentum (not reference)."
        ),
    )
    phase: FloatSequence = ROField(
        default=[0.0, 0.0], max_length=2, description="Photon E-field phase (x,y)."
    )
    px: float = ROField(default=0.0)
    py: float = ROField(default=0.0)
    pz: float = ROField(default=0.0)
    s: float = ROField(default=0.0, description="Longitudinal position")
    species: str = ROField(default="", description="positron$, proton$, etc.")
    spin: FloatSequence = ROField(default=[0.0, 0.0, 0.0], max_length=3, description="Spin.")
    state: str = ROField(
        default="", description="alive$, lost$, lost_neg_x_aperture$, lost_pz$, etc."
    )
    t: float = ROField(
        default=0.0, description="Absolute time (not relative to reference). Note: Quad precision!"
    )
    x: float = ROField(default=0.0)
    y: float = ROField(default=0.0)
    z: float = ROField(default=0.0)


class ElementPhotonBase(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:photon 1 base`, for example.

    Attributes
    ----------
    has_material : bool
    has_pixel : bool
    """

    _tao_command_: ClassVar[str] = "ele_photon"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}

    has_material: bool = ROField(default=False, attr="has_material", tao_name="has#material")
    has_pixel: bool = ROField(default=False, attr="has_pixel", tao_name="has#pixel")


class ElementPhotonCurvature(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:photon 1 curvature`, for example.

    Attributes
    ----------
    elliptical_curvature : sequence of floats
    spherical_curvature : float
    xy_0 : sequence of floats
    xy_1 : sequence of floats
    xy_2 : sequence of floats
    xy_3 : sequence of floats
    xy_4 : sequence of floats
    xy_5 : sequence of floats
    xy_6 : sequence of floats
    """

    _tao_command_: ClassVar[str] = "ele_photon"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}

    elliptical_curvature: FloatSequence = Field(default_factory=list, max_length=3)
    spherical_curvature: float = Field(default=0.0)
    xy_0: FloatSequence = Field(default_factory=list, max_length=7, attr="xy_0", tao_name="xy(0,:)")
    xy_1: FloatSequence = Field(default_factory=list, max_length=7, attr="xy_1", tao_name="xy(1,:)")
    xy_2: FloatSequence = Field(default_factory=list, max_length=7, attr="xy_2", tao_name="xy(2,:)")
    xy_3: FloatSequence = Field(default_factory=list, max_length=7, attr="xy_3", tao_name="xy(3,:)")
    xy_4: FloatSequence = Field(default_factory=list, max_length=7, attr="xy_4", tao_name="xy(4,:)")
    xy_5: FloatSequence = Field(default_factory=list, max_length=7, attr="xy_5", tao_name="xy(5,:)")
    xy_6: FloatSequence = Field(default_factory=list, max_length=7, attr="xy_6", tao_name="xy(6,:)")


class ElementPhotonMaterial(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:photon 2 material`, for example.

    Attributes
    ----------
    f0_m1 : complex or None
    f0_m2 : complex
    f_h : complex
    f_hbar : complex
    sqrt_f_h_f_hbar : complex
    """

    _tao_command_: ClassVar[str] = "ele_photon"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}

    f0_m1: complex | None = ROField(default=None, attr="f0_m1", tao_name="F0_m1")
    f0_m2: complex = ROField(default=0j, attr="f0_m2", tao_name="F0_m2")
    f_h: complex = ROField(default=0j, attr="f_h", tao_name="F_H")
    f_hbar: complex = ROField(default=0j, attr="f_hbar", tao_name="F_Hbar")
    sqrt_f_h_f_hbar: complex = ROField(
        default=0j, attr="sqrt_f_h_f_hbar", tao_name="Sqrt(F_H*F_Hbar)"
    )


class ElementTwiss(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:twiss 1`, for example.

    Attributes
    ----------
    alpha_a : float
    alpha_b : float
    beta_a : float
    beta_b : float
    eta_a : float
    eta_b : float
    eta_x : float
    eta_y : float
    etap_a : float
    etap_b : float
    etap_x : float
    etap_y : float
    gamma_a : float
    gamma_b : float
    mode_flip : bool
    phi_a : float
    phi_b : float
    """

    _tao_command_: ClassVar[str] = "ele_twiss"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}

    alpha_a: float = ROField(default=0.0)
    alpha_b: float = ROField(default=0.0)
    beta_a: float = ROField(default=0.0)
    beta_b: float = ROField(default=0.0)
    eta_a: float = ROField(default=0.0)
    eta_b: float = ROField(default=0.0)
    eta_x: float = ROField(default=0.0)
    eta_y: float = ROField(default=0.0)
    etap_a: float = ROField(default=0.0)
    etap_b: float = ROField(default=0.0)
    etap_x: float = ROField(default=0.0)
    etap_y: float = ROField(default=0.0)
    gamma_a: float = ROField(default=0.0)
    gamma_b: float = ROField(default=0.0)
    mode_flip: bool = ROField(default=False)
    phi_a: float = ROField(default=0.0)
    phi_b: float = ROField(default=0.0)


class ElementWakeBase(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:wake P3 base`, for example.

    Attributes
    ----------
    has_lr_mode : bool
    has_sr_long : bool
    has_sr_trans : bool
    lr_amp_scale : float
        Wake amplitude scale factor.
    lr_freq_spread : float
        Random frequency spread of long range modes.
    lr_self_wake_on : bool
        Long range self-wake used in tracking?
    lr_time_scale : float
        time scale factor.
    sr_amp_scale : float
        Wake amplitude scale factor.
    sr_scale_with_length : bool
        Scale wake with element length?
    sr_z_max : float
        Max allowable z value. 0-> ignore
    sr_z_scale : float
        z-distance scale factor.
    """

    _tao_command_: ClassVar[str] = "ele_wake"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}

    has_lr_mode: bool = ROField(default=False, attr="has_lr_mode", tao_name="has#lr_mode")
    has_sr_long: bool = ROField(default=False, attr="has_sr_long", tao_name="has#sr_long")
    has_sr_trans: bool = ROField(default=False, attr="has_sr_trans", tao_name="has#sr_trans")
    lr_amp_scale: float = Field(
        default=1.0,
        description="Wake amplitude scale factor.",
        attr="lr_amp_scale",
        tao_name="lr%amp_scale",
    )
    lr_freq_spread: float = Field(
        default=0.0,
        description="Random frequency spread of long range modes.",
        attr="lr_freq_spread",
        tao_name="lr%freq_spread",
    )
    lr_self_wake_on: bool = Field(
        default=True,
        description="Long range self-wake used in tracking?",
        attr="lr_self_wake_on",
        tao_name="lr%self_wake_on",
    )
    lr_time_scale: float = Field(
        default=1.0,
        description="time scale factor.",
        attr="lr_time_scale",
        tao_name="lr%time_scale",
    )
    sr_amp_scale: float = Field(
        default=1.0,
        description="Wake amplitude scale factor.",
        attr="sr_amp_scale",
        tao_name="sr%amp_scale",
    )
    sr_scale_with_length: bool = Field(
        default=True,
        description="Scale wake with element length?",
        attr="sr_scale_with_length",
        tao_name="sr%scale_with_length",
    )
    sr_z_max: float = Field(
        default=0.0,
        description="Max allowable z value. 0-> ignore",
        attr="sr_z_max",
        tao_name="sr%z_max",
    )
    sr_z_scale: float = Field(
        default=1.0,
        description="z-distance scale factor.",
        attr="sr_z_scale",
        tao_name="sr%z_scale",
    )


class ElementWakeSrLong(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:wake P3 sr_long`, for example.

    Attributes
    ----------
    z_ref : float
    """

    _tao_command_: ClassVar[str] = "ele_wake"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}

    z_ref: float = Field(default=0.0)


class ElementWakeSrTrans(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:wake P3 sr_long`, for example.

    Attributes
    ----------
    z_ref : float
    """

    _tao_command_: ClassVar[str] = "ele_wake"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}

    z_ref: float = Field(default=0.0)


class ElementWall3DBase(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:wall3d 1 1 base`, for example.

    Attributes
    ----------
    clear_material : str or None
    ele_anchor_pt : str
        anchor_beginning$, anchor_center$, or anchor_end$
    name : str
        Identifying name
    opaque_material : str or None
    superimpose : bool or None
        Can overlap another wall
    thickness : float or None
        Material thickness.
    """

    _tao_command_: ClassVar[str] = "ele_wall3d"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}

    clear_material: str | None = Field(default=None)
    ele_anchor_pt: str = Field(
        default="", description="anchor_beginning$, anchor_center$, or anchor_end$"
    )
    name: str = Field(default="", description="Identifying name")
    opaque_material: str | None = Field(default=None)
    superimpose: bool | None = Field(default=None, description="Can overlap another wall")
    thickness: float | None = Field(default=None, description="Material thickness.")


class ElementWall3DTable_Data(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:wall3d 1 1 table`, for example.

    Attributes
    ----------
    j : int
    radius_x : float
    radius_y : float
    tilt : float
    x : float
    y : float
    """

    _tao_command_: ClassVar[str] = "ele_wall3d"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}

    j: int = ROField(default=0)
    radius_x: float = ROField(default=0.0)
    radius_y: float = ROField(default=0.0)
    tilt: float = ROField(default=0.0)
    x: float = ROField(default=0.0)
    y: float = ROField(default=0.0)


class ElementWall3DTable(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:wall3d 1 1 table`, for example.

    Attributes
    ----------
    data : ElementWall3DTable_Data
        Structure which corresponds to Tao `pipe ele:wall3d 1 1 table`, for example.
    r0 : sequence of floats
        Center of section Section-to-section spline interpolation of the center of the
        section
    s : float
        Longitudinal position
    section : int
    vertex : int
    wall3d_section_type : str
    """

    _tao_command_: ClassVar[str] = "ele_wall3d"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}

    data: Sequence[ElementWall3DTable_Data] = ROField(
        default_factory=list,
        description="Structure which corresponds to Tao `pipe ele:wall3d 1 1 table`, for example.",
    )
    r0: FloatSequence = Field(
        default=[0.0, 0.0],
        max_length=2,
        description=(
            "Center of section Section-to-section spline interpolation of the center of the section"
        ),
    )
    s: float = Field(default=0.0, description="Longitudinal position")
    section: int = ROField(default=0)
    vertex: int = ROField(default=0)
    wall3d_section_type: str = ROField(
        default="", attr="wall3d_section_type", tao_name="wall3d_section^type"
    )


class SpaceChargeCom(TaoSettableModel):
    """
    Structure which corresponds to Tao `pipe space_charge_com`, for example.

    Attributes
    ----------
    abs_tol_tracking : float
        Absolute tolerance for tracking.
    beam_chamber_height : float
        Used in shielding calculation.
    cathode_strength_cutoff : float
        Cutoff for the cathode field calc.
    csr3d_mesh_size : sequence of integers
        Gird size for CSR.
    diagnostic_output_file : str
        If non-blank write a diagnostic (EG wake) file
    ds_track_step : float
        CSR tracking step size
    dt_track_step : float
        Time Runge kutta initial step.
    lsc_kick_transverse_dependence : bool
    lsc_sigma_cutoff : float
        Cutoff for the 1-dim longitudinal SC calc. If a bin sigma is < cutoff *
        sigma_ave then ignore.
    n_bin : int
        Number of bins used
    n_shield_images : int
        Chamber wall shielding. 0 = no shielding.
    particle_bin_span : int
        Longitudinal particle length / dz_bin
    particle_sigma_cutoff : float
        3D SC calc cutoff for particles with (x,y,z) position far from the center.
        Negative or zero means ignore.
    rel_tol_tracking : float
        Relative tolerance for tracking.
    sc_min_in_bin : int
        Minimum number of particles in a bin for sigmas to be valid.
    space_charge_mesh_size : sequence of integers
        Gird size for fft_3d space charge calc.
    """

    _tao_command_: ClassVar[str] = "space_charge_com"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()

    abs_tol_tracking: float = Field(default=1e-10, description="Absolute tolerance for tracking.")
    beam_chamber_height: float = Field(default=0.0, description="Used in shielding calculation.")
    cathode_strength_cutoff: float = Field(
        default=0.01, description="Cutoff for the cathode field calc."
    )
    csr3d_mesh_size: IntSequence = Field(
        default=[32, 32, 64], max_length=3, description="Gird size for CSR."
    )
    diagnostic_output_file: str = Field(
        default="", description="If non-blank write a diagnostic (EG wake) file"
    )
    ds_track_step: float = Field(default=0.0, description="CSR tracking step size")
    dt_track_step: float = Field(default=1e-12, description="Time Runge kutta initial step.")
    lsc_kick_transverse_dependence: bool = Field(default=False)
    lsc_sigma_cutoff: float = Field(
        default=0.1,
        description=(
            "Cutoff for the 1-dim longitudinal SC calc. If a bin sigma is < cutoff * "
            "sigma_ave then ignore."
        ),
    )
    n_bin: int = Field(default=0, description="Number of bins used")
    n_shield_images: int = Field(default=0, description="Chamber wall shielding. 0 = no shielding.")
    particle_bin_span: int = Field(default=2, description="Longitudinal particle length / dz_bin")
    particle_sigma_cutoff: float = Field(
        default=-1.0,
        description=(
            "3D SC calc cutoff for particles with (x,y,z) position far from the center. "
            "Negative or zero means ignore."
        ),
    )
    rel_tol_tracking: float = Field(default=1e-08, description="Relative tolerance for tracking.")
    sc_min_in_bin: int = Field(
        default=10, description="Minimum number of particles in a bin for sigmas to be valid."
    )
    space_charge_mesh_size: IntSequence = Field(
        default=[32, 32, 64], max_length=3, description="Gird size for fft_3d space charge calc."
    )


class TaoGlobal(TaoSettableModel):
    """
    Structure which corresponds to Tao `pipe global`, for example.

    Attributes
    ----------
    beam_timer_on : bool
        For timing the beam tracking calculation.
    box_plots : bool
        For debugging plot layout issues.
    bunch_to_plot : int
        Which bunch to plot
    concatenate_maps : bool
        False => tracking using DA.
    de_lm_step_ratio : float
        Scaling for step sizes between DE and LM optimizers.
    de_var_to_population_factor : float
        DE population = max(n_var*factor, 20)
    debug_on : bool
        For debugging.
    delta_e_chrom : float
        Delta E used from chrom calc.
    derivative_recalc : bool
        Recalc before each optimizer run?
    derivative_uses_design : bool
        Derivative calc uses design lattice instead of model?
    disable_smooth_line_calc : bool
        Global disable of the smooth line calculation.
    dmerit_stop_value : float
        Fractional Merit change below which an optimizer will stop.
    draw_curve_off_scale_warn : bool
        Display warning on graphs?
    external_plotting : bool
        Used with matplotlib and gui.
    label_keys : bool
        For lat_layout plots
    label_lattice_elements : bool
        For lat_layout plots
    lattice_calc_on : bool
        Turn on/off beam and single particle calculations.
    lm_opt_deriv_reinit : float
        Reinit derivative matrix cutoff
    lmdif_eps : float
        Tollerance for lmdif optimizer.
    lmdif_negligible_merit : float
    merit_stop_value : float
        Merit value below which an optimizer will stop.
    n_opti_cycles : int
        Number of optimization cycles
    n_opti_loops : int
        Number of optimization loops
    n_threads : int
        Number of OpenMP threads for parallel calculations.
    n_top10_merit : int
        Number of top merit constraints to print.
    only_limit_opt_vars : bool
        Only apply limits to variables used in optimization.
    opt_match_auto_recalc : bool
        Set recalc = True for match elements before each cycle?
    opt_with_base : bool
        Use base data in optimization?
    opt_with_ref : bool
        Use reference data in optimization?
    opti_write_var_file : bool
        "run" command writes var_out_file
    optimizer : str
        optimizer to use.
    optimizer_allow_user_abort : bool
        See Tao manual for more details.
    optimizer_var_limit_warn : bool
        Warn when vars reach a limit with optimization.
    phase_units : str
        Phase units on output.
    plot_on : bool
        Do plotting?
    print_command : str
    random_engine : str
        Non-beam random number engine
    random_gauss_converter : str
        Non-beam
    random_seed : int
        Use system clock by default
    random_sigma_cutoff : float
        Cut-off in sigmas.
    rf_on : bool
        RFcavities on or off? Does not affect lcavities.
    srdt_gen_n_slices : int
        Number times to slice elements for summation RDT calculation
    srdt_sxt_n_slices : int
        Number times to slice sextupoles for summation RDT calculation
    srdt_use_cache : bool
        Create cache for SRDT calculations.  Can use lots of memory if srdt_*_n_slices
        large.
    stop_on_error : bool
        For debugging: False prevents tao from exiting on an error.
    svd_cutoff : float
        SVD singular value cutoff.
    svd_retreat_on_merit_increase : bool
    symbol_import : bool
        Import symbols from lattice file(s)?
    track_type : str
        or 'beam'
    unstable_penalty : float
        Used in unstable_ring datum merit calculation.
    var_limits_on : bool
        Respect the variable limits?
    var_out_file : str
    wait_for_cr_in_single_mode : bool
        For use with a python GUI.
    """

    _tao_command_: ClassVar[str] = "tao_global"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()

    beam_timer_on: bool = Field(
        default=False, description="For timing the beam tracking calculation."
    )
    box_plots: bool = Field(default=False, description="For debugging plot layout issues.")
    bunch_to_plot: int = Field(default=1, description="Which bunch to plot")
    concatenate_maps: bool = Field(default=False, description="False => tracking using DA.")
    de_lm_step_ratio: float = Field(
        default=1.0, description="Scaling for step sizes between DE and LM optimizers."
    )
    de_var_to_population_factor: float = Field(
        default=5.0, description="DE population = max(n_var*factor, 20)"
    )
    debug_on: bool = Field(default=False, description="For debugging.")
    delta_e_chrom: float = Field(default=0.0, description="Delta E used from chrom calc.")
    derivative_recalc: bool = Field(default=True, description="Recalc before each optimizer run?")
    derivative_uses_design: bool = Field(
        default=False, description="Derivative calc uses design lattice instead of model?"
    )
    disable_smooth_line_calc: bool = Field(
        default=False, description="Global disable of the smooth line calculation."
    )
    dmerit_stop_value: float = Field(
        default=0.0, description="Fractional Merit change below which an optimizer will stop."
    )
    draw_curve_off_scale_warn: bool = Field(default=True, description="Display warning on graphs?")
    external_plotting: bool = ROField(default=False, description="Used with matplotlib and gui.")
    label_keys: bool = Field(default=True, description="For lat_layout plots")
    label_lattice_elements: bool = Field(default=True, description="For lat_layout plots")
    lattice_calc_on: bool = Field(
        default=True, description="Turn on/off beam and single particle calculations."
    )
    lm_opt_deriv_reinit: float = Field(default=-1.0, description="Reinit derivative matrix cutoff")
    lmdif_eps: float = Field(default=1e-12, description="Tollerance for lmdif optimizer.")
    lmdif_negligible_merit: float = Field(default=1e-30)
    merit_stop_value: float = Field(
        default=0.0, description="Merit value below which an optimizer will stop."
    )
    n_opti_cycles: int = Field(default=20, description="Number of optimization cycles")
    n_opti_loops: int = Field(default=1, description="Number of optimization loops")
    n_threads: int = Field(
        default=1, description="Number of OpenMP threads for parallel calculations."
    )
    n_top10_merit: int = Field(default=10, description="Number of top merit constraints to print.")
    only_limit_opt_vars: bool = Field(
        default=False, description="Only apply limits to variables used in optimization."
    )
    opt_match_auto_recalc: bool = Field(
        default=False, description="Set recalc = True for match elements before each cycle?"
    )
    opt_with_base: bool = Field(default=False, description="Use base data in optimization?")
    opt_with_ref: bool = Field(default=False, description="Use reference data in optimization?")
    opti_write_var_file: bool = Field(default=True, description="'run' command writes var_out_file")
    optimizer: str = Field(default="lm", description="optimizer to use.")
    optimizer_allow_user_abort: bool = Field(
        default=True, description="See Tao manual for more details."
    )
    optimizer_var_limit_warn: bool = Field(
        default=True, description="Warn when vars reach a limit with optimization."
    )
    phase_units: str = Field(default="", description="Phase units on output.")
    plot_on: bool = Field(default=True, description="Do plotting?")
    print_command: str = Field(default="lpr")
    random_engine: str = Field(default="", description="Non-beam random number engine")
    random_gauss_converter: str = Field(default="", description="Non-beam")
    random_seed: int = Field(default=-1, description="Use system clock by default")
    random_sigma_cutoff: float = Field(default=-1.0, description="Cut-off in sigmas.")
    rf_on: bool = Field(
        default=True, description="RFcavities on or off? Does not affect lcavities."
    )
    srdt_gen_n_slices: int = Field(
        default=10, description="Number times to slice elements for summation RDT calculation"
    )
    srdt_sxt_n_slices: int = Field(
        default=20, description="Number times to slice sextupoles for summation RDT calculation"
    )
    srdt_use_cache: bool = Field(
        default=True,
        description=(
            "Create cache for SRDT calculations.  Can use lots of memory if srdt_*_n_slices large."
        ),
    )
    stop_on_error: bool = Field(
        default=True, description="For debugging: False prevents tao from exiting on an error."
    )
    svd_cutoff: float = Field(default=1e-05, description="SVD singular value cutoff.")
    svd_retreat_on_merit_increase: bool = Field(default=True)
    symbol_import: bool = Field(default=False, description="Import symbols from lattice file(s)?")
    track_type: str = Field(default="single", description="or 'beam'")
    unstable_penalty: float = Field(
        default=0.001, description="Used in unstable_ring datum merit calculation."
    )
    var_limits_on: bool = Field(default=True, description="Respect the variable limits?")
    var_out_file: str = Field(default="var#.out")
    wait_for_cr_in_single_mode: bool = Field(
        default=False, description="For use with a python GUI."
    )
