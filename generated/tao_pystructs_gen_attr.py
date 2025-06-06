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
    Any,
    ClassVar,
    Literal,
    cast,
)

import numpy as np
import pydantic
from genesis.version4.types import _check_equality
from pytao import Tao
from rich.pretty import pretty_repr
from typing_extensions import Self

# TODO: this code is partly duplicated with TaoSettableModel; refactor
# TODO: reduce reliance on [internal-ish] lume-genesis API (genesis.version4.types._check_equality)
logger = logging.getLogger(__name__)


class TaoAttributesModel(
    pydantic.BaseModel,
    validate_assignment=True,
    extra="allow",  # NOTE: differs from the rest of pystructs
):
    """
    A helper base class which allows for creating/updating an instance with Tao objects.
    """

    _tao_command_: ClassVar[str] = ""
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {"which": "model"}
    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()

    command_args: dict[str, int | float | str] = pydantic.Field(
        default_factory=dict,
        frozen=True,
        description="Arguments used for the pytao command to generate this structure",
        repr=False,
    )

    def query(self, tao: Tao) -> Self:
        """Query Tao again to generate a new instance of this model."""
        if "ele_id" not in self.command_args:
            raise ValueError(
                f"This class instance does not have an associated 'ele_id'.  Did you create it using "
                f"{type(self).__name__}.from_tao(tao, ele_id=...)?"
            )

        return self.from_tao(tao, **self.command_args)

    @classmethod
    def from_tao(cls: type[Self], tao: Tao, ele_id: str, **kwargs) -> Self:
        """
        Create this structure by querying Tao for its current values.

        Parameters
        ----------
        tao : Tao
        **kwargs :
            Keyword arguments to pass to the relevant ``tao`` command.
        """
        cmd_kwargs = dict(cls._tao_command_default_args_)
        cmd_kwargs["ele_id"] = ele_id
        cmd_kwargs.update(**kwargs)

        key = generalattributes_to_key[cls]
        data: dict[str, Any] = tao.ele_gen_attribs(**cmd_kwargs)
        return cls(command_args=cmd_kwargs, key=key, **data)

    def __eq__(self, other) -> bool:
        return _check_equality(self, other)

    def __repr__(self):
        return pretty_repr(self)

    @property
    def settable_fields(self) -> list[str]:
        """Names of all 'settable' (modifiable) fields."""
        return [
            attr
            for attr, field_info in self.model_fields.items()
            if not field_info.frozen and attr not in {"key"}
        ]

    @property
    def set_commands(self) -> list[str]:
        """
        Get all Tao 'set' commands to apply this configuration.

        Returns
        -------
        list of str
        """
        return self.get_set_commands(tao=None)

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

    def get_set_commands(self, tao: Tao | None = None) -> list[str]:
        cmds = []

        try:
            ele_id = self.command_args["ele_id"]
        except KeyError:
            raise ValueError(
                f"This class instance does not have an associated 'ele_id'.  Did you create it using "
                f"{type(self).__name__}.from_tao(tao, ele_id=...)?"
            )

        if tao is not None:
            attrs = self._get_changed_attributes(tao)
        else:
            attrs = self._all_attributes_to_set

        for attr, index, value in attrs:
            value = getattr(self, attr)

            if attr in self._tao_skip_if_0_ and value == 0:
                continue

            fld = self.model_fields[attr]
            name = fld.serialization_alias or attr
            if index is None:
                cmds.append(f"set element {ele_id} {name} = {value}")
            else:
                cmds.append(f"set element {ele_id} {name}({index + 1}) = {value}")

        return cmds

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


class AB_multipoleAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 1`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    delta_ref_time : float
    e_tot : float
    field_master : bool
    L : float
    lord_status : str
    offset_moves_aperture : bool
    p0c : float
    ref_time_start : float
    slave_status : str
    tilt : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_offset_tot : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_offset_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["AB_multipole"]  # = "AB_multipole"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    field_master: bool = Field(default=False)
    L: float = ROField(default=0.0)
    lord_status: str = ROField(default="")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    slave_status: str = ROField(default="")
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class AC_KickerAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 2`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    bl_hkick : float
    bl_vkick : float
    csr_ds_step : float
    delta_ref_time : float
    ds_step : float
    e_tot : float
    field_master : bool
    fringe_at : str
    fringe_type : str
    hkick : float
    integrator_order : int
    interpolation : str
    L : float
    lord_pad1 : float
    lord_pad2 : float
    lord_status : str
    num_steps : int
    offset_moves_aperture : bool
    p0c : float
    phi0_multipass : float
    r0_elec : float
    r0_mag : float
    ref_time_start : float
    slave_status : str
    spin_fringe_on : bool
    static_linear_map : bool
    t_offset : float
    tilt : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    vkick : float
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["AC_Kicker"]  # = "AC_Kicker"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    bl_hkick: float = ROField(default=0.0, attr="bl_hkick", tao_name="BL_HKICK")
    bl_vkick: float = ROField(default=0.0, attr="bl_vkick", tao_name="BL_VKICK")
    csr_ds_step: float = ROField(default=0.0, attr="csr_ds_step", tao_name="CSR_DS_STEP")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    ds_step: float = ROField(default=0.0, attr="ds_step", tao_name="DS_STEP")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    field_master: bool = Field(default=False)
    fringe_at: str = ROField(default="", attr="fringe_at", tao_name="FRINGE_AT")
    fringe_type: str = ROField(default="", attr="fringe_type", tao_name="FRINGE_TYPE")
    hkick: float = ROField(default=0.0, attr="hkick", tao_name="HKICK")
    integrator_order: int = ROField(default=0, attr="integrator_order", tao_name="INTEGRATOR_ORDER")
    interpolation: str = ROField(default="", attr="interpolation", tao_name="INTERPOLATION")
    L: float = ROField(default=0.0)
    lord_pad1: float = ROField(default=0.0, attr="lord_pad1", tao_name="LORD_PAD1")
    lord_pad2: float = ROField(default=0.0, attr="lord_pad2", tao_name="LORD_PAD2")
    lord_status: str = ROField(default="")
    num_steps: int = ROField(default=0, attr="num_steps", tao_name="NUM_STEPS")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    phi0_multipass: float = ROField(default=0.0, attr="phi0_multipass", tao_name="PHI0_MULTIPASS")
    r0_elec: float = ROField(default=0.0, attr="r0_elec", tao_name="R0_ELEC")
    r0_mag: float = ROField(default=0.0, attr="r0_mag", tao_name="R0_MAG")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    slave_status: str = ROField(default="")
    spin_fringe_on: bool = ROField(default=False, attr="spin_fringe_on", tao_name="SPIN_FRINGE_ON")
    static_linear_map: bool = ROField(
        default=False, attr="static_linear_map", tao_name="STATIC_LINEAR_MAP"
    )
    t_offset: float = ROField(default=0.0, attr="t_offset", tao_name="T_OFFSET")
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    vkick: float = ROField(default=0.0, attr="vkick", tao_name="VKICK")
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class BeamBeamAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 5`, for example.

    Attributes
    ----------
    alpha_a_strong : float
    alpha_b_strong : float
    aperture_at : str
    aperture_type : str
    bbi_constant : float
    beta_a_strong : float
    beta_b_strong : float
    bs_field : float
    charge : float
    cmat_11 : float
    cmat_12 : float
    cmat_21 : float
    cmat_22 : float
    crab_tilt : float
    crab_x1 : float
    crab_x2 : float
    crab_x3 : float
    crab_x4 : float
    crab_x5 : float
    delta_ref_time : float
    e_tot : float
    e_tot_strong : float
    ks : float
    L : float
    lord_status : str
    n_particle : float
    n_slice : int
    offset_moves_aperture : bool
    p0c : float
    ref_time_start : float
    repetition_frequency : float
    s_twiss_ref : float
    sig_x : float
    sig_y : float
    sig_z : float
    slave_status : str
    species_strong : str
    tilt : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_crossing : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["BeamBeam"]  # = "BeamBeam"

    alpha_a_strong: float = ROField(default=0.0, attr="alpha_a_strong", tao_name="ALPHA_A_STRONG")
    alpha_b_strong: float = ROField(default=0.0, attr="alpha_b_strong", tao_name="ALPHA_B_STRONG")
    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    bbi_constant: float = ROField(default=0.0, attr="bbi_constant", tao_name="BBI_CONSTANT")
    beta_a_strong: float = ROField(default=0.0, attr="beta_a_strong", tao_name="BETA_A_STRONG")
    beta_b_strong: float = ROField(default=0.0, attr="beta_b_strong", tao_name="BETA_B_STRONG")
    bs_field: float = ROField(default=0.0, attr="bs_field", tao_name="BS_FIELD")
    charge: float = ROField(default=0.0, attr="charge", tao_name="CHARGE")
    cmat_11: float = ROField(default=0.0, attr="cmat_11", tao_name="CMAT_11")
    cmat_12: float = ROField(default=0.0, attr="cmat_12", tao_name="CMAT_12")
    cmat_21: float = ROField(default=0.0, attr="cmat_21", tao_name="CMAT_21")
    cmat_22: float = ROField(default=0.0, attr="cmat_22", tao_name="CMAT_22")
    crab_tilt: float = ROField(default=0.0, attr="crab_tilt", tao_name="CRAB_TILT")
    crab_x1: float = ROField(default=0.0, attr="crab_x1", tao_name="CRAB_X1")
    crab_x2: float = ROField(default=0.0, attr="crab_x2", tao_name="CRAB_X2")
    crab_x3: float = ROField(default=0.0, attr="crab_x3", tao_name="CRAB_X3")
    crab_x4: float = ROField(default=0.0, attr="crab_x4", tao_name="CRAB_X4")
    crab_x5: float = ROField(default=0.0, attr="crab_x5", tao_name="CRAB_X5")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    e_tot_strong: float = ROField(default=0.0, attr="e_tot_strong", tao_name="E_TOT_STRONG")
    ks: float = ROField(default=0.0, attr="ks", tao_name="KS")
    L: float = ROField(default=0.0)
    lord_status: str = ROField(default="")
    n_particle: float = ROField(default=0.0, attr="n_particle", tao_name="N_PARTICLE")
    n_slice: int = ROField(default=0, attr="n_slice", tao_name="N_SLICE")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    repetition_frequency: float = ROField(
        default=0.0, attr="repetition_frequency", tao_name="REPETITION_FREQUENCY"
    )
    s_twiss_ref: float = ROField(default=0.0, attr="s_twiss_ref", tao_name="S_TWISS_REF")
    sig_x: float = ROField(default=0.0, attr="sig_x", tao_name="SIG_X")
    sig_y: float = ROField(default=0.0, attr="sig_y", tao_name="SIG_Y")
    sig_z: float = ROField(default=0.0, attr="sig_z", tao_name="SIG_Z")
    slave_status: str = ROField(default="")
    species_strong: str = ROField(default="", attr="species_strong", tao_name="SPECIES_STRONG")
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_crossing: float = ROField(default=0.0, attr="z_crossing", tao_name="Z_CROSSING")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class Beginning_EleAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 0`, for example.

    Attributes
    ----------
    cmat_11 : float
    cmat_12 : float
    cmat_21 : float
    cmat_22 : float
    e_tot : float
    e_tot_start : float
    inherit_from_fork : bool
    lord_status : str
    mode_flip : bool
    p0c : float
    p0c_start : float
    slave_status : str
    spin_dn_dpz_x : float
    spin_dn_dpz_y : float
    spin_dn_dpz_z : float
    units : dict[str, str]
        Per-attribute unit information.
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Beginning_Ele"]  # = "Beginning_Ele"

    cmat_11: float = ROField(default=0.0, attr="cmat_11", tao_name="CMAT_11")
    cmat_12: float = ROField(default=0.0, attr="cmat_12", tao_name="CMAT_12")
    cmat_21: float = ROField(default=0.0, attr="cmat_21", tao_name="CMAT_21")
    cmat_22: float = ROField(default=0.0, attr="cmat_22", tao_name="CMAT_22")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    e_tot_start: float = ROField(default=0.0, attr="e_tot_start", tao_name="E_TOT_START")
    inherit_from_fork: bool = ROField(
        default=False, attr="inherit_from_fork", tao_name="INHERIT_FROM_FORK"
    )
    lord_status: str = ROField(default="")
    mode_flip: bool = ROField(default=False, attr="mode_flip", tao_name="MODE_FLIP")
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    p0c_start: float = ROField(default=0.0, attr="p0c_start", tao_name="P0C_START")
    slave_status: str = ROField(default="")
    spin_dn_dpz_x: float = ROField(default=0.0, attr="spin_dn_dpz_x", tao_name="SPIN_DN_DPZ_X")
    spin_dn_dpz_y: float = ROField(default=0.0, attr="spin_dn_dpz_y", tao_name="SPIN_DN_DPZ_Y")
    spin_dn_dpz_z: float = ROField(default=0.0, attr="spin_dn_dpz_z", tao_name="SPIN_DN_DPZ_Z")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )


class Crab_CavityAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 6`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    bl_hkick : float
    bl_vkick : float
    csr_ds_step : float
    delta_ref_time : float
    ds_step : float
    e_tot : float
    field_master : bool
    gradient : float
    harmon : float
    harmon_master : bool
    hkick : float
    integrator_order : int
    L : float
    lord_pad1 : float
    lord_pad2 : float
    lord_status : str
    num_steps : int
    offset_moves_aperture : bool
    p0c : float
    phi0 : float
    phi0_multipass : float
    ref_time_start : float
    rf_frequency : float
    rf_wavelength : float
    slave_status : str
    static_linear_map : bool
    tilt : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    vkick : float
    voltage : float
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Crab_Cavity"]  # = "Crab_Cavity"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    bl_hkick: float = ROField(default=0.0, attr="bl_hkick", tao_name="BL_HKICK")
    bl_vkick: float = ROField(default=0.0, attr="bl_vkick", tao_name="BL_VKICK")
    csr_ds_step: float = ROField(default=0.0, attr="csr_ds_step", tao_name="CSR_DS_STEP")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    ds_step: float = ROField(default=0.0, attr="ds_step", tao_name="DS_STEP")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    field_master: bool = Field(default=False)
    gradient: float = ROField(default=0.0, attr="gradient", tao_name="GRADIENT")
    harmon: float = ROField(default=0.0, attr="harmon", tao_name="HARMON")
    harmon_master: bool = ROField(default=False, attr="harmon_master", tao_name="HARMON_MASTER")
    hkick: float = ROField(default=0.0, attr="hkick", tao_name="HKICK")
    integrator_order: int = ROField(default=0, attr="integrator_order", tao_name="INTEGRATOR_ORDER")
    L: float = ROField(default=0.0)
    lord_pad1: float = ROField(default=0.0, attr="lord_pad1", tao_name="LORD_PAD1")
    lord_pad2: float = ROField(default=0.0, attr="lord_pad2", tao_name="LORD_PAD2")
    lord_status: str = ROField(default="")
    num_steps: int = ROField(default=0, attr="num_steps", tao_name="NUM_STEPS")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    phi0: float = ROField(default=0.0, attr="phi0", tao_name="PHI0")
    phi0_multipass: float = ROField(default=0.0, attr="phi0_multipass", tao_name="PHI0_MULTIPASS")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    rf_frequency: float = ROField(default=0.0, attr="rf_frequency", tao_name="RF_FREQUENCY")
    rf_wavelength: float = ROField(default=0.0, attr="rf_wavelength", tao_name="RF_WAVELENGTH")
    slave_status: str = ROField(default="")
    static_linear_map: bool = ROField(
        default=False, attr="static_linear_map", tao_name="STATIC_LINEAR_MAP"
    )
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    vkick: float = ROField(default=0.0, attr="vkick", tao_name="VKICK")
    voltage: float = ROField(default=0.0, attr="voltage", tao_name="VOLTAGE")
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class CrystalAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 2`, for example.

    Attributes
    ----------
    alpha_angle : float
    aperture_at : str
    aperture_type : str
    b_param : float
    bragg_angle : float
    bragg_angle_in : float
    bragg_angle_out : float
    d_spacing : float
    darwin_width_pi : float
    darwin_width_sigma : float
    dbragg_angle_de : float
    delta_ref_time : float
    e_tot : float
    graze_angle_in : float
    graze_angle_out : float
    is_mosaic : bool
    L : float
    lord_status : str
    mosaic_angle_rms_in_plane : float
    mosaic_angle_rms_out_plane : float
    mosaic_diffraction_num : int
    mosaic_thickness : float
    offset_moves_aperture : bool
    p0c : float
    pendellosung_period_pi : float
    pendellosung_period_sigma : float
    psi_angle : float
    ref_cap_gamma : float
    ref_orbit_follows : str
    ref_tilt : float
    ref_tilt_tot : float
    ref_time_start : float
    ref_wavelength : float
    slave_status : str
    thickness : float
    tilt : float
    tilt_corr : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    use_reflectivity_table : bool
    v_unitcell : float
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Crystal"]  # = "Crystal"

    alpha_angle: float = ROField(default=0.0, attr="alpha_angle", tao_name="ALPHA_ANGLE")
    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    b_param: float = ROField(default=0.0, attr="b_param", tao_name="B_PARAM")
    bragg_angle: float = ROField(default=0.0, attr="bragg_angle", tao_name="BRAGG_ANGLE")
    bragg_angle_in: float = ROField(default=0.0, attr="bragg_angle_in", tao_name="BRAGG_ANGLE_IN")
    bragg_angle_out: float = ROField(
        default=0.0, attr="bragg_angle_out", tao_name="BRAGG_ANGLE_OUT"
    )
    d_spacing: float = ROField(default=0.0, attr="d_spacing", tao_name="D_SPACING")
    darwin_width_pi: float = ROField(
        default=0.0, attr="darwin_width_pi", tao_name="DARWIN_WIDTH_PI"
    )
    darwin_width_sigma: float = ROField(
        default=0.0, attr="darwin_width_sigma", tao_name="DARWIN_WIDTH_SIGMA"
    )
    dbragg_angle_de: float = ROField(
        default=0.0, attr="dbragg_angle_de", tao_name="DBRAGG_ANGLE_DE"
    )
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    graze_angle_in: float = ROField(default=0.0, attr="graze_angle_in", tao_name="GRAZE_ANGLE_IN")
    graze_angle_out: float = ROField(
        default=0.0, attr="graze_angle_out", tao_name="GRAZE_ANGLE_OUT"
    )
    is_mosaic: bool = ROField(default=False, attr="is_mosaic", tao_name="IS_MOSAIC")
    L: float = ROField(default=0.0)
    lord_status: str = ROField(default="")
    mosaic_angle_rms_in_plane: float = ROField(
        default=0.0, attr="mosaic_angle_rms_in_plane", tao_name="MOSAIC_ANGLE_RMS_IN_PLANE"
    )
    mosaic_angle_rms_out_plane: float = ROField(
        default=0.0, attr="mosaic_angle_rms_out_plane", tao_name="MOSAIC_ANGLE_RMS_OUT_PLANE"
    )
    mosaic_diffraction_num: int = ROField(
        default=0, attr="mosaic_diffraction_num", tao_name="MOSAIC_DIFFRACTION_NUM"
    )
    mosaic_thickness: float = ROField(
        default=0.0, attr="mosaic_thickness", tao_name="MOSAIC_THICKNESS"
    )
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    pendellosung_period_pi: float = ROField(
        default=0.0, attr="pendellosung_period_pi", tao_name="PENDELLOSUNG_PERIOD_PI"
    )
    pendellosung_period_sigma: float = ROField(
        default=0.0, attr="pendellosung_period_sigma", tao_name="PENDELLOSUNG_PERIOD_SIGMA"
    )
    psi_angle: float = ROField(default=0.0, attr="psi_angle", tao_name="PSI_ANGLE")
    ref_cap_gamma: float = ROField(default=0.0, attr="ref_cap_gamma", tao_name="REF_CAP_GAMMA")
    ref_orbit_follows: str = ROField(
        default="", attr="ref_orbit_follows", tao_name="REF_ORBIT_FOLLOWS"
    )
    ref_tilt: float = ROField(default=0.0, attr="ref_tilt", tao_name="REF_TILT")
    ref_tilt_tot: float = ROField(default=0.0, attr="ref_tilt_tot", tao_name="REF_TILT_TOT")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    ref_wavelength: float = ROField(default=0.0, attr="ref_wavelength", tao_name="REF_WAVELENGTH")
    slave_status: str = ROField(default="")
    thickness: float = ROField(default=0.0, attr="thickness", tao_name="THICKNESS")
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_corr: float = ROField(default=0.0, attr="tilt_corr", tao_name="TILT_CORR")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    use_reflectivity_table: bool = ROField(
        default=False, attr="use_reflectivity_table", tao_name="USE_REFLECTIVITY_TABLE"
    )
    v_unitcell: float = ROField(default=0.0, attr="v_unitcell", tao_name="V_UNITCELL")
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class DetectorAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 2`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    crunch : float
    crunch_calib : float
    de_eta_meas : float
    delta_ref_time : float
    e_tot : float
    L : float
    lord_status : str
    n_sample : float
    noise : float
    offset_moves_aperture : bool
    osc_amplitude : float
    p0c : float
    ref_time_start : float
    slave_status : str
    tilt : float
    tilt_calib : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    x1_limit : float
    x2_limit : float
    x_dispersion_calib : float
    x_dispersion_err : float
    x_gain_calib : float
    x_gain_err : float
    x_offset : float
    x_offset_calib : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_dispersion_calib : float
    y_dispersion_err : float
    y_gain_calib : float
    y_gain_err : float
    y_offset : float
    y_offset_calib : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Detector"]  # = "Detector"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    crunch: float = ROField(default=0.0, attr="crunch", tao_name="CRUNCH")
    crunch_calib: float = ROField(default=0.0, attr="crunch_calib", tao_name="CRUNCH_CALIB")
    de_eta_meas: float = ROField(default=0.0, attr="de_eta_meas", tao_name="DE_ETA_MEAS")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    L: float = ROField(default=0.0)
    lord_status: str = ROField(default="")
    n_sample: float = ROField(default=0.0, attr="n_sample", tao_name="N_SAMPLE")
    noise: float = ROField(default=0.0, attr="noise", tao_name="NOISE")
    offset_moves_aperture: bool = Field(default=False)
    osc_amplitude: float = ROField(default=0.0, attr="osc_amplitude", tao_name="OSC_AMPLITUDE")
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    slave_status: str = ROField(default="")
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_calib: float = ROField(default=0.0, attr="tilt_calib", tao_name="TILT_CALIB")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_dispersion_calib: float = ROField(
        default=0.0, attr="x_dispersion_calib", tao_name="X_DISPERSION_CALIB"
    )
    x_dispersion_err: float = ROField(
        default=0.0, attr="x_dispersion_err", tao_name="X_DISPERSION_ERR"
    )
    x_gain_calib: float = ROField(default=0.0, attr="x_gain_calib", tao_name="X_GAIN_CALIB")
    x_gain_err: float = ROField(default=0.0, attr="x_gain_err", tao_name="X_GAIN_ERR")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_calib: float = ROField(default=0.0, attr="x_offset_calib", tao_name="X_OFFSET_CALIB")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_dispersion_calib: float = ROField(
        default=0.0, attr="y_dispersion_calib", tao_name="Y_DISPERSION_CALIB"
    )
    y_dispersion_err: float = ROField(
        default=0.0, attr="y_dispersion_err", tao_name="Y_DISPERSION_ERR"
    )
    y_gain_calib: float = ROField(default=0.0, attr="y_gain_calib", tao_name="Y_GAIN_CALIB")
    y_gain_err: float = ROField(default=0.0, attr="y_gain_err", tao_name="Y_GAIN_ERR")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_calib: float = ROField(default=0.0, attr="y_offset_calib", tao_name="Y_OFFSET_CALIB")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class DriftAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 7`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    csr_ds_step : float
    delta_ref_time : float
    ds_step : float
    e_tot : float
    integrator_order : int
    L : float
    lord_pad1 : float
    lord_pad2 : float
    lord_status : str
    num_steps : int
    offset_moves_aperture : bool
    p0c : float
    ref_time_start : float
    slave_status : str
    static_linear_map : bool
    tilt : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Drift"]  # = "Drift"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    csr_ds_step: float = ROField(default=0.0, attr="csr_ds_step", tao_name="CSR_DS_STEP")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    ds_step: float = ROField(default=0.0, attr="ds_step", tao_name="DS_STEP")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    integrator_order: int = ROField(default=0, attr="integrator_order", tao_name="INTEGRATOR_ORDER")
    L: float = ROField(default=0.0)
    lord_pad1: float = ROField(default=0.0, attr="lord_pad1", tao_name="LORD_PAD1")
    lord_pad2: float = ROField(default=0.0, attr="lord_pad2", tao_name="LORD_PAD2")
    lord_status: str = ROField(default="")
    num_steps: int = ROField(default=0, attr="num_steps", tao_name="NUM_STEPS")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    slave_status: str = ROField(default="")
    static_linear_map: bool = ROField(
        default=False, attr="static_linear_map", tao_name="STATIC_LINEAR_MAP"
    )
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class E_GunAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 1`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    autoscale_amplitude : bool
    autoscale_phase : bool
    csr_ds_step : float
    delta_ref_time : float
    ds_step : float
    dt_max : float
    e_tot : float
    emit_fraction : float
    field_autoscale : float
    fringe_at : str
    fringe_type : str
    gradient : float
    gradient_err : float
    gradient_tot : float
    integrator_order : int
    L : float
    lord_pad1 : float
    lord_pad2 : float
    lord_status : str
    num_steps : int
    offset_moves_aperture : bool
    p0c : float
    phi0 : float
    phi0_autoscale : float
    phi0_err : float
    ref_time_start : float
    rf_frequency : float
    rf_wavelength : float
    slave_status : str
    spin_fringe_on : bool
    static_linear_map : bool
    tilt : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    voltage : float
    voltage_err : float
    voltage_tot : float
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["E_Gun"]  # = "E_Gun"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    autoscale_amplitude: bool = ROField(
        default=False, attr="autoscale_amplitude", tao_name="AUTOSCALE_AMPLITUDE"
    )
    autoscale_phase: bool = ROField(
        default=False, attr="autoscale_phase", tao_name="AUTOSCALE_PHASE"
    )
    csr_ds_step: float = ROField(default=0.0, attr="csr_ds_step", tao_name="CSR_DS_STEP")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    ds_step: float = ROField(default=0.0, attr="ds_step", tao_name="DS_STEP")
    dt_max: float = ROField(default=0.0, attr="dt_max", tao_name="DT_MAX")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    emit_fraction: float = ROField(default=0.0, attr="emit_fraction", tao_name="EMIT_FRACTION")
    field_autoscale: float = ROField(
        default=0.0, attr="field_autoscale", tao_name="FIELD_AUTOSCALE"
    )
    fringe_at: str = ROField(default="", attr="fringe_at", tao_name="FRINGE_AT")
    fringe_type: str = ROField(default="", attr="fringe_type", tao_name="FRINGE_TYPE")
    gradient: float = ROField(default=0.0, attr="gradient", tao_name="GRADIENT")
    gradient_err: float = ROField(default=0.0, attr="gradient_err", tao_name="GRADIENT_ERR")
    gradient_tot: float = ROField(default=0.0, attr="gradient_tot", tao_name="GRADIENT_TOT")
    integrator_order: int = ROField(default=0, attr="integrator_order", tao_name="INTEGRATOR_ORDER")
    L: float = ROField(default=0.0)
    lord_pad1: float = ROField(default=0.0, attr="lord_pad1", tao_name="LORD_PAD1")
    lord_pad2: float = ROField(default=0.0, attr="lord_pad2", tao_name="LORD_PAD2")
    lord_status: str = ROField(default="")
    num_steps: int = ROField(default=0, attr="num_steps", tao_name="NUM_STEPS")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    phi0: float = ROField(default=0.0, attr="phi0", tao_name="PHI0")
    phi0_autoscale: float = ROField(default=0.0, attr="phi0_autoscale", tao_name="PHI0_AUTOSCALE")
    phi0_err: float = ROField(default=0.0, attr="phi0_err", tao_name="PHI0_ERR")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    rf_frequency: float = ROField(default=0.0, attr="rf_frequency", tao_name="RF_FREQUENCY")
    rf_wavelength: float = ROField(default=0.0, attr="rf_wavelength", tao_name="RF_WAVELENGTH")
    slave_status: str = ROField(default="")
    spin_fringe_on: bool = ROField(default=False, attr="spin_fringe_on", tao_name="SPIN_FRINGE_ON")
    static_linear_map: bool = ROField(
        default=False, attr="static_linear_map", tao_name="STATIC_LINEAR_MAP"
    )
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    voltage: float = ROField(default=0.0, attr="voltage", tao_name="VOLTAGE")
    voltage_err: float = ROField(default=0.0, attr="voltage_err", tao_name="VOLTAGE_ERR")
    voltage_tot: float = ROField(default=0.0, attr="voltage_tot", tao_name="VOLTAGE_TOT")
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class ECollimatorAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 8`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    bl_hkick : float
    bl_vkick : float
    csr_ds_step : float
    delta_ref_time : float
    ds_step : float
    e_tot : float
    fringe_at : str
    fringe_type : str
    hkick : float
    integrator_order : int
    L : float
    lord_pad1 : float
    lord_pad2 : float
    lord_status : str
    num_steps : int
    offset_moves_aperture : bool
    p0c : float
    px_aperture_center : float
    px_aperture_width2 : float
    py_aperture_center : float
    py_aperture_width2 : float
    pz_aperture_center : float
    pz_aperture_width2 : float
    ref_time_start : float
    slave_status : str
    spin_fringe_on : bool
    static_linear_map : bool
    tilt : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    vkick : float
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_aperture_center : float
    z_aperture_width2 : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["ECollimator"]  # = "ECollimator"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    bl_hkick: float = ROField(default=0.0, attr="bl_hkick", tao_name="BL_HKICK")
    bl_vkick: float = ROField(default=0.0, attr="bl_vkick", tao_name="BL_VKICK")
    csr_ds_step: float = ROField(default=0.0, attr="csr_ds_step", tao_name="CSR_DS_STEP")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    ds_step: float = ROField(default=0.0, attr="ds_step", tao_name="DS_STEP")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    fringe_at: str = ROField(default="", attr="fringe_at", tao_name="FRINGE_AT")
    fringe_type: str = ROField(default="", attr="fringe_type", tao_name="FRINGE_TYPE")
    hkick: float = ROField(default=0.0, attr="hkick", tao_name="HKICK")
    integrator_order: int = ROField(default=0, attr="integrator_order", tao_name="INTEGRATOR_ORDER")
    L: float = ROField(default=0.0)
    lord_pad1: float = ROField(default=0.0, attr="lord_pad1", tao_name="LORD_PAD1")
    lord_pad2: float = ROField(default=0.0, attr="lord_pad2", tao_name="LORD_PAD2")
    lord_status: str = ROField(default="")
    num_steps: int = ROField(default=0, attr="num_steps", tao_name="NUM_STEPS")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    px_aperture_center: float = ROField(
        default=0.0, attr="px_aperture_center", tao_name="PX_APERTURE_CENTER"
    )
    px_aperture_width2: float = ROField(
        default=0.0, attr="px_aperture_width2", tao_name="PX_APERTURE_WIDTH2"
    )
    py_aperture_center: float = ROField(
        default=0.0, attr="py_aperture_center", tao_name="PY_APERTURE_CENTER"
    )
    py_aperture_width2: float = ROField(
        default=0.0, attr="py_aperture_width2", tao_name="PY_APERTURE_WIDTH2"
    )
    pz_aperture_center: float = ROField(
        default=0.0, attr="pz_aperture_center", tao_name="PZ_APERTURE_CENTER"
    )
    pz_aperture_width2: float = ROField(
        default=0.0, attr="pz_aperture_width2", tao_name="PZ_APERTURE_WIDTH2"
    )
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    slave_status: str = ROField(default="")
    spin_fringe_on: bool = ROField(default=False, attr="spin_fringe_on", tao_name="SPIN_FRINGE_ON")
    static_linear_map: bool = ROField(
        default=False, attr="static_linear_map", tao_name="STATIC_LINEAR_MAP"
    )
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    vkick: float = ROField(default=0.0, attr="vkick", tao_name="VKICK")
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_aperture_center: float = ROField(
        default=0.0, attr="z_aperture_center", tao_name="Z_APERTURE_CENTER"
    )
    z_aperture_width2: float = ROField(
        default=0.0, attr="z_aperture_width2", tao_name="Z_APERTURE_WIDTH2"
    )
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class ELSeparatorAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 9`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    csr_ds_step : float
    delta_ref_time : float
    ds_step : float
    e_field : float
    e_tot : float
    field_master : bool
    fringe_at : str
    fringe_type : str
    gap : float
    hkick : float
    integrator_order : int
    L : float
    lord_pad1 : float
    lord_pad2 : float
    lord_status : str
    num_steps : int
    offset_moves_aperture : bool
    p0c : float
    ptc_canonical_coords : bool
    r0_elec : float
    r0_mag : float
    ref_time_start : float
    slave_status : str
    spin_fringe_on : bool
    static_linear_map : bool
    tilt : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    vkick : float
    voltage : float
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["ELSeparator"]  # = "ELSeparator"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    csr_ds_step: float = ROField(default=0.0, attr="csr_ds_step", tao_name="CSR_DS_STEP")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    ds_step: float = ROField(default=0.0, attr="ds_step", tao_name="DS_STEP")
    e_field: float = ROField(default=0.0, attr="e_field", tao_name="E_FIELD")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    field_master: bool = Field(default=False)
    fringe_at: str = ROField(default="", attr="fringe_at", tao_name="FRINGE_AT")
    fringe_type: str = ROField(default="", attr="fringe_type", tao_name="FRINGE_TYPE")
    gap: float = ROField(default=0.0, attr="gap", tao_name="GAP")
    hkick: float = ROField(default=0.0, attr="hkick", tao_name="HKICK")
    integrator_order: int = ROField(default=0, attr="integrator_order", tao_name="INTEGRATOR_ORDER")
    L: float = ROField(default=0.0)
    lord_pad1: float = ROField(default=0.0, attr="lord_pad1", tao_name="LORD_PAD1")
    lord_pad2: float = ROField(default=0.0, attr="lord_pad2", tao_name="LORD_PAD2")
    lord_status: str = ROField(default="")
    num_steps: int = ROField(default=0, attr="num_steps", tao_name="NUM_STEPS")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    ptc_canonical_coords: bool = ROField(
        default=False, attr="ptc_canonical_coords", tao_name="PTC_CANONICAL_COORDS"
    )
    r0_elec: float = ROField(default=0.0, attr="r0_elec", tao_name="R0_ELEC")
    r0_mag: float = ROField(default=0.0, attr="r0_mag", tao_name="R0_MAG")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    slave_status: str = ROField(default="")
    spin_fringe_on: bool = ROField(default=False, attr="spin_fringe_on", tao_name="SPIN_FRINGE_ON")
    static_linear_map: bool = ROField(
        default=False, attr="static_linear_map", tao_name="STATIC_LINEAR_MAP"
    )
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    vkick: float = ROField(default=0.0, attr="vkick", tao_name="VKICK")
    voltage: float = ROField(default=0.0, attr="voltage", tao_name="VOLTAGE")
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class EM_FieldAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 11`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    autoscale_amplitude : bool
    autoscale_phase : bool
    constant_ref_energy : bool
    csr_ds_step : float
    delta_ref_time : float
    ds_step : float
    e_tot : float
    e_tot_start : float
    field_autoscale : float
    fringe_at : str
    fringe_type : str
    integrator_order : int
    L : float
    lord_pad1 : float
    lord_pad2 : float
    lord_status : str
    num_steps : int
    offset_moves_aperture : bool
    p0c : float
    p0c_start : float
    phi0 : float
    phi0_autoscale : float
    phi0_err : float
    polarity : float
    ptc_canonical_coords : bool
    ref_time_start : float
    rf_frequency : float
    rf_wavelength : float
    slave_status : str
    spin_fringe_on : bool
    static_linear_map : bool
    tilt : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["EM_Field"]  # = "EM_Field"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    autoscale_amplitude: bool = ROField(
        default=False, attr="autoscale_amplitude", tao_name="AUTOSCALE_AMPLITUDE"
    )
    autoscale_phase: bool = ROField(
        default=False, attr="autoscale_phase", tao_name="AUTOSCALE_PHASE"
    )
    constant_ref_energy: bool = ROField(
        default=False, attr="constant_ref_energy", tao_name="CONSTANT_REF_ENERGY"
    )
    csr_ds_step: float = ROField(default=0.0, attr="csr_ds_step", tao_name="CSR_DS_STEP")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    ds_step: float = ROField(default=0.0, attr="ds_step", tao_name="DS_STEP")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    e_tot_start: float = ROField(default=0.0, attr="e_tot_start", tao_name="E_TOT_START")
    field_autoscale: float = ROField(
        default=0.0, attr="field_autoscale", tao_name="FIELD_AUTOSCALE"
    )
    fringe_at: str = ROField(default="", attr="fringe_at", tao_name="FRINGE_AT")
    fringe_type: str = ROField(default="", attr="fringe_type", tao_name="FRINGE_TYPE")
    integrator_order: int = ROField(default=0, attr="integrator_order", tao_name="INTEGRATOR_ORDER")
    L: float = ROField(default=0.0)
    lord_pad1: float = ROField(default=0.0, attr="lord_pad1", tao_name="LORD_PAD1")
    lord_pad2: float = ROField(default=0.0, attr="lord_pad2", tao_name="LORD_PAD2")
    lord_status: str = ROField(default="")
    num_steps: int = ROField(default=0, attr="num_steps", tao_name="NUM_STEPS")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    p0c_start: float = ROField(default=0.0, attr="p0c_start", tao_name="P0C_START")
    phi0: float = ROField(default=0.0, attr="phi0", tao_name="PHI0")
    phi0_autoscale: float = ROField(default=0.0, attr="phi0_autoscale", tao_name="PHI0_AUTOSCALE")
    phi0_err: float = ROField(default=0.0, attr="phi0_err", tao_name="PHI0_ERR")
    polarity: float = ROField(default=0.0, attr="polarity", tao_name="POLARITY")
    ptc_canonical_coords: bool = ROField(
        default=False, attr="ptc_canonical_coords", tao_name="PTC_CANONICAL_COORDS"
    )
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    rf_frequency: float = ROField(default=0.0, attr="rf_frequency", tao_name="RF_FREQUENCY")
    rf_wavelength: float = ROField(default=0.0, attr="rf_wavelength", tao_name="RF_WAVELENGTH")
    slave_status: str = ROField(default="")
    spin_fringe_on: bool = ROField(default=False, attr="spin_fringe_on", tao_name="SPIN_FRINGE_ON")
    static_linear_map: bool = ROField(
        default=False, attr="static_linear_map", tao_name="STATIC_LINEAR_MAP"
    )
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class FiducialAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 13`, for example.

    Attributes
    ----------
    delta_ref_time : float
    dphi_origin : float
    dpsi_origin : float
    dtheta_origin : float
    dx_origin : float
    dy_origin : float
    dz_origin : float
    e_tot : float
    L : float
    lord_status : str
    origin_ele_ref_pt : str
    p0c : float
    ref_time_start : float
    slave_status : str
    units : dict[str, str]
        Per-attribute unit information.
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Fiducial"]  # = "Fiducial"

    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    dphi_origin: float = ROField(default=0.0, attr="dphi_origin", tao_name="DPHI_ORIGIN")
    dpsi_origin: float = ROField(default=0.0, attr="dpsi_origin", tao_name="DPSI_ORIGIN")
    dtheta_origin: float = ROField(default=0.0, attr="dtheta_origin", tao_name="DTHETA_ORIGIN")
    dx_origin: float = ROField(default=0.0, attr="dx_origin", tao_name="DX_ORIGIN")
    dy_origin: float = ROField(default=0.0, attr="dy_origin", tao_name="DY_ORIGIN")
    dz_origin: float = ROField(default=0.0, attr="dz_origin", tao_name="DZ_ORIGIN")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    L: float = ROField(default=0.0)
    lord_status: str = ROField(default="")
    origin_ele_ref_pt: str = ROField(
        default="", attr="origin_ele_ref_pt", tao_name="ORIGIN_ELE_REF_PT"
    )
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    slave_status: str = ROField(default="")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )


class Floor_ShiftAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 14`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    delta_ref_time : float
    downstream_ele_dir : int
    e_tot : float
    L : float
    lord_status : str
    offset_moves_aperture : bool
    origin_ele_ref_pt : str
    p0c : float
    ref_time_start : float
    slave_status : str
    tilt : float
    units : dict[str, str]
        Per-attribute unit information.
    upstream_ele_dir : int
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_pitch : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_pitch : float
    z_offset : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Floor_Shift"]  # = "Floor_Shift"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    downstream_ele_dir: int = ROField(
        default=0, attr="downstream_ele_dir", tao_name="DOWNSTREAM_ELE_DIR"
    )
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    L: float = ROField(default=0.0)
    lord_status: str = ROField(default="")
    offset_moves_aperture: bool = Field(default=False)
    origin_ele_ref_pt: str = ROField(
        default="", attr="origin_ele_ref_pt", tao_name="ORIGIN_ELE_REF_PT"
    )
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    slave_status: str = ROField(default="")
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    upstream_ele_dir: int = ROField(default=0, attr="upstream_ele_dir", tao_name="UPSTREAM_ELE_DIR")
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")


class ForkAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 2`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    delta_ref_time : float
    direction : int
    e_tot : float
    ix_to_branch : int
    ix_to_element : int
    L : float
    lord_status : str
    new_branch : bool
    offset_moves_aperture : bool
    p0c : float
    ref_time_start : float
    slave_status : str
    units : dict[str, str]
        Per-attribute unit information.
    x1_limit : float
    x2_limit : float
    y1_limit : float
    y2_limit : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Fork"]  # = "Fork"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    direction: int = ROField(default=0, attr="direction", tao_name="DIRECTION")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    ix_to_branch: int = ROField(default=0, attr="ix_to_branch", tao_name="IX_TO_BRANCH")
    ix_to_element: int = ROField(default=0, attr="ix_to_element", tao_name="IX_TO_ELEMENT")
    L: float = ROField(default=0.0)
    lord_status: str = ROField(default="")
    new_branch: bool = ROField(default=False, attr="new_branch", tao_name="NEW_BRANCH")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    slave_status: str = ROField(default="")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")


class GirderAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 24`, for example.

    Attributes
    ----------
    dphi_origin : float
    dpsi_origin : float
    dtheta_origin : float
    dx_origin : float
    dy_origin : float
    dz_origin : float
    L : float
    lord_status : str
    origin_ele_ref_pt : str
    ref_tilt : float
    ref_tilt_tot : float
    slave_status : str
    tilt : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    x_offset : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y_offset : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Girder"]  # = "Girder"

    dphi_origin: float = ROField(default=0.0, attr="dphi_origin", tao_name="DPHI_ORIGIN")
    dpsi_origin: float = ROField(default=0.0, attr="dpsi_origin", tao_name="DPSI_ORIGIN")
    dtheta_origin: float = ROField(default=0.0, attr="dtheta_origin", tao_name="DTHETA_ORIGIN")
    dx_origin: float = ROField(default=0.0, attr="dx_origin", tao_name="DX_ORIGIN")
    dy_origin: float = ROField(default=0.0, attr="dy_origin", tao_name="DY_ORIGIN")
    dz_origin: float = ROField(default=0.0, attr="dz_origin", tao_name="DZ_ORIGIN")
    L: float = ROField(default=0.0)
    lord_status: str = ROField(default="")
    origin_ele_ref_pt: str = ROField(
        default="", attr="origin_ele_ref_pt", tao_name="ORIGIN_ELE_REF_PT"
    )
    ref_tilt: float = ROField(default=0.0, attr="ref_tilt", tao_name="REF_TILT")
    ref_tilt_tot: float = ROField(default=0.0, attr="ref_tilt_tot", tao_name="REF_TILT_TOT")
    slave_status: str = ROField(default="")
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class GKickerAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 15`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    delta_ref_time : float
    e_tot : float
    lord_status : str
    offset_moves_aperture : bool
    p0c : float
    px_kick : float
    py_kick : float
    pz_kick : float
    ref_time_start : float
    slave_status : str
    units : dict[str, str]
        Per-attribute unit information.
    x1_limit : float
    x2_limit : float
    x_kick : float
    y1_limit : float
    y2_limit : float
    y_kick : float
    z_kick : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["GKicker"]  # = "GKicker"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    lord_status: str = ROField(default="")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    px_kick: float = ROField(default=0.0, attr="px_kick", tao_name="PX_KICK")
    py_kick: float = ROField(default=0.0, attr="py_kick", tao_name="PY_KICK")
    pz_kick: float = ROField(default=0.0, attr="pz_kick", tao_name="PZ_KICK")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    slave_status: str = ROField(default="")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_kick: float = ROField(default=0.0, attr="x_kick", tao_name="X_KICK")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_kick: float = ROField(default=0.0, attr="y_kick", tao_name="Y_KICK")
    z_kick: float = ROField(default=0.0, attr="z_kick", tao_name="Z_KICK")


class GroupAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 17`, for example.

    Attributes
    ----------
    gang : bool
    interpolation : str
    lord_status : str
    slave_status : str
    units : dict[str, str]
        Per-attribute unit information.
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Group"]  # = "Group"

    gang: bool = ROField(default=False, attr="gang", tao_name="GANG")
    interpolation: str = ROField(default="", attr="interpolation", tao_name="INTERPOLATION")
    lord_status: str = ROField(default="")
    slave_status: str = ROField(default="")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )


class HKickerAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 16`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    bl_kick : float
    csr_ds_step : float
    delta_ref_time : float
    ds_step : float
    e_tot : float
    field_master : bool
    fringe_at : str
    fringe_type : str
    integrator_order : int
    kick : float
    L : float
    lord_pad1 : float
    lord_pad2 : float
    lord_status : str
    num_steps : int
    offset_moves_aperture : bool
    p0c : float
    ptc_canonical_coords : bool
    ref_time_start : float
    slave_status : str
    spin_fringe_on : bool
    static_linear_map : bool
    tilt : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["HKicker"]  # = "HKicker"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    bl_kick: float = ROField(default=0.0, attr="bl_kick", tao_name="BL_KICK")
    csr_ds_step: float = ROField(default=0.0, attr="csr_ds_step", tao_name="CSR_DS_STEP")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    ds_step: float = ROField(default=0.0, attr="ds_step", tao_name="DS_STEP")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    field_master: bool = Field(default=False)
    fringe_at: str = ROField(default="", attr="fringe_at", tao_name="FRINGE_AT")
    fringe_type: str = ROField(default="", attr="fringe_type", tao_name="FRINGE_TYPE")
    integrator_order: int = ROField(default=0, attr="integrator_order", tao_name="INTEGRATOR_ORDER")
    kick: float = ROField(default=0.0, attr="kick", tao_name="KICK")
    L: float = ROField(default=0.0)
    lord_pad1: float = ROField(default=0.0, attr="lord_pad1", tao_name="LORD_PAD1")
    lord_pad2: float = ROField(default=0.0, attr="lord_pad2", tao_name="LORD_PAD2")
    lord_status: str = ROField(default="")
    num_steps: int = ROField(default=0, attr="num_steps", tao_name="NUM_STEPS")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    ptc_canonical_coords: bool = ROField(
        default=False, attr="ptc_canonical_coords", tao_name="PTC_CANONICAL_COORDS"
    )
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    slave_status: str = ROField(default="")
    spin_fringe_on: bool = ROField(default=False, attr="spin_fringe_on", tao_name="SPIN_FRINGE_ON")
    static_linear_map: bool = ROField(
        default=False, attr="static_linear_map", tao_name="STATIC_LINEAR_MAP"
    )
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class InstrumentAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 17`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    bl_hkick : float
    bl_vkick : float
    crunch : float
    crunch_calib : float
    csr_ds_step : float
    de_eta_meas : float
    delta_ref_time : float
    ds_step : float
    e_tot : float
    fringe_at : str
    fringe_type : str
    hkick : float
    integrator_order : int
    L : float
    lord_pad1 : float
    lord_pad2 : float
    lord_status : str
    n_sample : float
    noise : float
    num_steps : int
    offset_moves_aperture : bool
    osc_amplitude : float
    p0c : float
    ref_time_start : float
    slave_status : str
    spin_fringe_on : bool
    static_linear_map : bool
    tilt : float
    tilt_calib : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    vkick : float
    x1_limit : float
    x2_limit : float
    x_dispersion_calib : float
    x_dispersion_err : float
    x_gain_calib : float
    x_gain_err : float
    x_offset : float
    x_offset_calib : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_dispersion_calib : float
    y_dispersion_err : float
    y_gain_calib : float
    y_gain_err : float
    y_offset : float
    y_offset_calib : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Instrument"]  # = "Instrument"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    bl_hkick: float = ROField(default=0.0, attr="bl_hkick", tao_name="BL_HKICK")
    bl_vkick: float = ROField(default=0.0, attr="bl_vkick", tao_name="BL_VKICK")
    crunch: float = ROField(default=0.0, attr="crunch", tao_name="CRUNCH")
    crunch_calib: float = ROField(default=0.0, attr="crunch_calib", tao_name="CRUNCH_CALIB")
    csr_ds_step: float = ROField(default=0.0, attr="csr_ds_step", tao_name="CSR_DS_STEP")
    de_eta_meas: float = ROField(default=0.0, attr="de_eta_meas", tao_name="DE_ETA_MEAS")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    ds_step: float = ROField(default=0.0, attr="ds_step", tao_name="DS_STEP")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    fringe_at: str = ROField(default="", attr="fringe_at", tao_name="FRINGE_AT")
    fringe_type: str = ROField(default="", attr="fringe_type", tao_name="FRINGE_TYPE")
    hkick: float = ROField(default=0.0, attr="hkick", tao_name="HKICK")
    integrator_order: int = ROField(default=0, attr="integrator_order", tao_name="INTEGRATOR_ORDER")
    L: float = ROField(default=0.0)
    lord_pad1: float = ROField(default=0.0, attr="lord_pad1", tao_name="LORD_PAD1")
    lord_pad2: float = ROField(default=0.0, attr="lord_pad2", tao_name="LORD_PAD2")
    lord_status: str = ROField(default="")
    n_sample: float = ROField(default=0.0, attr="n_sample", tao_name="N_SAMPLE")
    noise: float = ROField(default=0.0, attr="noise", tao_name="NOISE")
    num_steps: int = ROField(default=0, attr="num_steps", tao_name="NUM_STEPS")
    offset_moves_aperture: bool = Field(default=False)
    osc_amplitude: float = ROField(default=0.0, attr="osc_amplitude", tao_name="OSC_AMPLITUDE")
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    slave_status: str = ROField(default="")
    spin_fringe_on: bool = ROField(default=False, attr="spin_fringe_on", tao_name="SPIN_FRINGE_ON")
    static_linear_map: bool = ROField(
        default=False, attr="static_linear_map", tao_name="STATIC_LINEAR_MAP"
    )
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_calib: float = ROField(default=0.0, attr="tilt_calib", tao_name="TILT_CALIB")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    vkick: float = ROField(default=0.0, attr="vkick", tao_name="VKICK")
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_dispersion_calib: float = ROField(
        default=0.0, attr="x_dispersion_calib", tao_name="X_DISPERSION_CALIB"
    )
    x_dispersion_err: float = ROField(
        default=0.0, attr="x_dispersion_err", tao_name="X_DISPERSION_ERR"
    )
    x_gain_calib: float = ROField(default=0.0, attr="x_gain_calib", tao_name="X_GAIN_CALIB")
    x_gain_err: float = ROField(default=0.0, attr="x_gain_err", tao_name="X_GAIN_ERR")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_calib: float = ROField(default=0.0, attr="x_offset_calib", tao_name="X_OFFSET_CALIB")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_dispersion_calib: float = ROField(
        default=0.0, attr="y_dispersion_calib", tao_name="Y_DISPERSION_CALIB"
    )
    y_dispersion_err: float = ROField(
        default=0.0, attr="y_dispersion_err", tao_name="Y_DISPERSION_ERR"
    )
    y_gain_calib: float = ROField(default=0.0, attr="y_gain_calib", tao_name="Y_GAIN_CALIB")
    y_gain_err: float = ROField(default=0.0, attr="y_gain_err", tao_name="Y_GAIN_ERR")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_calib: float = ROField(default=0.0, attr="y_offset_calib", tao_name="Y_OFFSET_CALIB")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class KickerAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 18`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    bl_hkick : float
    bl_vkick : float
    csr_ds_step : float
    delta_ref_time : float
    ds_step : float
    e_tot : float
    field_master : bool
    fringe_at : str
    fringe_type : str
    h_displace : float
    hkick : float
    integrator_order : int
    L : float
    lord_pad1 : float
    lord_pad2 : float
    lord_status : str
    num_steps : int
    offset_moves_aperture : bool
    p0c : float
    ptc_canonical_coords : bool
    r0_elec : float
    r0_mag : float
    ref_time_start : float
    slave_status : str
    spin_fringe_on : bool
    static_linear_map : bool
    tilt : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    v_displace : float
    vkick : float
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Kicker"]  # = "Kicker"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    bl_hkick: float = ROField(default=0.0, attr="bl_hkick", tao_name="BL_HKICK")
    bl_vkick: float = ROField(default=0.0, attr="bl_vkick", tao_name="BL_VKICK")
    csr_ds_step: float = ROField(default=0.0, attr="csr_ds_step", tao_name="CSR_DS_STEP")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    ds_step: float = ROField(default=0.0, attr="ds_step", tao_name="DS_STEP")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    field_master: bool = Field(default=False)
    fringe_at: str = ROField(default="", attr="fringe_at", tao_name="FRINGE_AT")
    fringe_type: str = ROField(default="", attr="fringe_type", tao_name="FRINGE_TYPE")
    h_displace: float = ROField(default=0.0, attr="h_displace", tao_name="H_DISPLACE")
    hkick: float = ROField(default=0.0, attr="hkick", tao_name="HKICK")
    integrator_order: int = ROField(default=0, attr="integrator_order", tao_name="INTEGRATOR_ORDER")
    L: float = ROField(default=0.0)
    lord_pad1: float = ROField(default=0.0, attr="lord_pad1", tao_name="LORD_PAD1")
    lord_pad2: float = ROField(default=0.0, attr="lord_pad2", tao_name="LORD_PAD2")
    lord_status: str = ROField(default="")
    num_steps: int = ROField(default=0, attr="num_steps", tao_name="NUM_STEPS")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    ptc_canonical_coords: bool = ROField(
        default=False, attr="ptc_canonical_coords", tao_name="PTC_CANONICAL_COORDS"
    )
    r0_elec: float = ROField(default=0.0, attr="r0_elec", tao_name="R0_ELEC")
    r0_mag: float = ROField(default=0.0, attr="r0_mag", tao_name="R0_MAG")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    slave_status: str = ROField(default="")
    spin_fringe_on: bool = ROField(default=False, attr="spin_fringe_on", tao_name="SPIN_FRINGE_ON")
    static_linear_map: bool = ROField(
        default=False, attr="static_linear_map", tao_name="STATIC_LINEAR_MAP"
    )
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    v_displace: float = ROField(default=0.0, attr="v_displace", tao_name="V_DISPLACE")
    vkick: float = ROField(default=0.0, attr="vkick", tao_name="VKICK")
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class LcavityAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 53`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    autoscale_amplitude : bool
    autoscale_phase : bool
    bl_hkick : float
    bl_vkick : float
    cavity_type : str
    coupler_angle : float
    coupler_at : str
    coupler_phase : float
    coupler_strength : float
    csr_ds_step : float
    delta_ref_time : float
    ds_step : float
    e_loss : float
    e_tot : float
    e_tot_start : float
    field_autoscale : float
    field_master : bool
    fringe_at : str
    fringe_type : str
    gradient : float
    gradient_err : float
    gradient_tot : float
    hkick : float
    integrator_order : int
    L : float
    l_active : float
    longitudinal_mode : int
    lord_pad1 : float
    lord_pad2 : float
    lord_status : str
    n_cell : int
    num_steps : int
    offset_moves_aperture : bool
    p0c : float
    p0c_start : float
    phi0 : float
    phi0_autoscale : float
    phi0_err : float
    phi0_multipass : float
    ref_time_start : float
    rf_frequency : float
    rf_wavelength : float
    slave_status : str
    spin_fringe_on : bool
    static_linear_map : bool
    tilt : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    vkick : float
    voltage : float
    voltage_err : float
    voltage_tot : float
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Lcavity"]  # = "Lcavity"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    autoscale_amplitude: bool = ROField(
        default=False, attr="autoscale_amplitude", tao_name="AUTOSCALE_AMPLITUDE"
    )
    autoscale_phase: bool = ROField(
        default=False, attr="autoscale_phase", tao_name="AUTOSCALE_PHASE"
    )
    bl_hkick: float = ROField(default=0.0, attr="bl_hkick", tao_name="BL_HKICK")
    bl_vkick: float = ROField(default=0.0, attr="bl_vkick", tao_name="BL_VKICK")
    cavity_type: str = ROField(default="", attr="cavity_type", tao_name="CAVITY_TYPE")
    coupler_angle: float = ROField(default=0.0, attr="coupler_angle", tao_name="COUPLER_ANGLE")
    coupler_at: str = ROField(default="", attr="coupler_at", tao_name="COUPLER_AT")
    coupler_phase: float = ROField(default=0.0, attr="coupler_phase", tao_name="COUPLER_PHASE")
    coupler_strength: float = ROField(
        default=0.0, attr="coupler_strength", tao_name="COUPLER_STRENGTH"
    )
    csr_ds_step: float = ROField(default=0.0, attr="csr_ds_step", tao_name="CSR_DS_STEP")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    ds_step: float = ROField(default=0.0, attr="ds_step", tao_name="DS_STEP")
    e_loss: float = ROField(default=0.0, attr="e_loss", tao_name="E_LOSS")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    e_tot_start: float = ROField(default=0.0, attr="e_tot_start", tao_name="E_TOT_START")
    field_autoscale: float = ROField(
        default=0.0, attr="field_autoscale", tao_name="FIELD_AUTOSCALE"
    )
    field_master: bool = Field(default=False)
    fringe_at: str = ROField(default="", attr="fringe_at", tao_name="FRINGE_AT")
    fringe_type: str = ROField(default="", attr="fringe_type", tao_name="FRINGE_TYPE")
    gradient: float = ROField(default=0.0, attr="gradient", tao_name="GRADIENT")
    gradient_err: float = ROField(default=0.0, attr="gradient_err", tao_name="GRADIENT_ERR")
    gradient_tot: float = ROField(default=0.0, attr="gradient_tot", tao_name="GRADIENT_TOT")
    hkick: float = ROField(default=0.0, attr="hkick", tao_name="HKICK")
    integrator_order: int = ROField(default=0, attr="integrator_order", tao_name="INTEGRATOR_ORDER")
    L: float = ROField(default=0.0)
    l_active: float = ROField(default=0.0, attr="l_active", tao_name="L_ACTIVE")
    longitudinal_mode: int = ROField(
        default=0, attr="longitudinal_mode", tao_name="LONGITUDINAL_MODE"
    )
    lord_pad1: float = ROField(default=0.0, attr="lord_pad1", tao_name="LORD_PAD1")
    lord_pad2: float = ROField(default=0.0, attr="lord_pad2", tao_name="LORD_PAD2")
    lord_status: str = ROField(default="")
    n_cell: int = ROField(default=0, attr="n_cell", tao_name="N_CELL")
    num_steps: int = ROField(default=0, attr="num_steps", tao_name="NUM_STEPS")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    p0c_start: float = ROField(default=0.0, attr="p0c_start", tao_name="P0C_START")
    phi0: float = ROField(default=0.0, attr="phi0", tao_name="PHI0")
    phi0_autoscale: float = ROField(default=0.0, attr="phi0_autoscale", tao_name="PHI0_AUTOSCALE")
    phi0_err: float = ROField(default=0.0, attr="phi0_err", tao_name="PHI0_ERR")
    phi0_multipass: float = ROField(default=0.0, attr="phi0_multipass", tao_name="PHI0_MULTIPASS")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    rf_frequency: float = ROField(default=0.0, attr="rf_frequency", tao_name="RF_FREQUENCY")
    rf_wavelength: float = ROField(default=0.0, attr="rf_wavelength", tao_name="RF_WAVELENGTH")
    slave_status: str = ROField(default="")
    spin_fringe_on: bool = ROField(default=False, attr="spin_fringe_on", tao_name="SPIN_FRINGE_ON")
    static_linear_map: bool = ROField(
        default=False, attr="static_linear_map", tao_name="STATIC_LINEAR_MAP"
    )
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    vkick: float = ROField(default=0.0, attr="vkick", tao_name="VKICK")
    voltage: float = ROField(default=0.0, attr="voltage", tao_name="VOLTAGE")
    voltage_err: float = ROField(default=0.0, attr="voltage_err", tao_name="VOLTAGE_ERR")
    voltage_tot: float = ROField(default=0.0, attr="voltage_tot", tao_name="VOLTAGE_TOT")
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class MarkerAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 56`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    crunch : float
    crunch_calib : float
    de_eta_meas : float
    delta_ref_time : float
    e_tot : float
    L : float
    lord_status : str
    n_sample : float
    noise : float
    offset_moves_aperture : bool
    osc_amplitude : float
    p0c : float
    ref_time_start : float
    slave_status : str
    tilt : float
    tilt_calib : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    x1_limit : float
    x2_limit : float
    x_dispersion_calib : float
    x_dispersion_err : float
    x_gain_calib : float
    x_gain_err : float
    x_offset : float
    x_offset_calib : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_dispersion_calib : float
    y_dispersion_err : float
    y_gain_calib : float
    y_gain_err : float
    y_offset : float
    y_offset_calib : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Marker"]  # = "Marker"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    crunch: float = ROField(default=0.0, attr="crunch", tao_name="CRUNCH")
    crunch_calib: float = ROField(default=0.0, attr="crunch_calib", tao_name="CRUNCH_CALIB")
    de_eta_meas: float = ROField(default=0.0, attr="de_eta_meas", tao_name="DE_ETA_MEAS")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    L: float = ROField(default=0.0)
    lord_status: str = ROField(default="")
    n_sample: float = ROField(default=0.0, attr="n_sample", tao_name="N_SAMPLE")
    noise: float = ROField(default=0.0, attr="noise", tao_name="NOISE")
    offset_moves_aperture: bool = Field(default=False)
    osc_amplitude: float = ROField(default=0.0, attr="osc_amplitude", tao_name="OSC_AMPLITUDE")
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    slave_status: str = ROField(default="")
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_calib: float = ROField(default=0.0, attr="tilt_calib", tao_name="TILT_CALIB")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_dispersion_calib: float = ROField(
        default=0.0, attr="x_dispersion_calib", tao_name="X_DISPERSION_CALIB"
    )
    x_dispersion_err: float = ROField(
        default=0.0, attr="x_dispersion_err", tao_name="X_DISPERSION_ERR"
    )
    x_gain_calib: float = ROField(default=0.0, attr="x_gain_calib", tao_name="X_GAIN_CALIB")
    x_gain_err: float = ROField(default=0.0, attr="x_gain_err", tao_name="X_GAIN_ERR")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_calib: float = ROField(default=0.0, attr="x_offset_calib", tao_name="X_OFFSET_CALIB")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_dispersion_calib: float = ROField(
        default=0.0, attr="y_dispersion_calib", tao_name="Y_DISPERSION_CALIB"
    )
    y_dispersion_err: float = ROField(
        default=0.0, attr="y_dispersion_err", tao_name="Y_DISPERSION_ERR"
    )
    y_gain_calib: float = ROField(default=0.0, attr="y_gain_calib", tao_name="Y_GAIN_CALIB")
    y_gain_err: float = ROField(default=0.0, attr="y_gain_err", tao_name="Y_GAIN_ERR")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_calib: float = ROField(default=0.0, attr="y_offset_calib", tao_name="Y_OFFSET_CALIB")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class MaskAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 1`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    delta_ref_time : float
    e_tot : float
    field_scale_factor : float
    lord_status : str
    mode : str
    offset_moves_aperture : bool
    p0c : float
    ref_time_start : float
    ref_wavelength : float
    slave_status : str
    tilt : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Mask"]  # = "Mask"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    field_scale_factor: float = ROField(
        default=0.0, attr="field_scale_factor", tao_name="FIELD_SCALE_FACTOR"
    )
    lord_status: str = ROField(default="")
    mode: str = ROField(default="", attr="mode", tao_name="MODE")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    ref_wavelength: float = ROField(default=0.0, attr="ref_wavelength", tao_name="REF_WAVELENGTH")
    slave_status: str = ROField(default="")
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class MatchAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 19`, for example.

    Attributes
    ----------
    alpha_a0 : float
    alpha_a1 : float
    alpha_b0 : float
    alpha_b1 : float
    aperture_at : str
    aperture_type : str
    beta_a0 : float
    beta_a1 : float
    beta_b0 : float
    beta_b1 : float
    c11_mat0 : float
    c11_mat1 : float
    c12_mat0 : float
    c12_mat1 : float
    c21_mat0 : float
    c21_mat1 : float
    c22_mat0 : float
    c22_mat1 : float
    delta_ref_time : float
    delta_time : float
    dphi_a : float
    dphi_b : float
    e_tot : float
    eta_x0 : float
    eta_x1 : float
    eta_y0 : float
    eta_y1 : float
    etap_x0 : float
    etap_x1 : float
    etap_y0 : float
    etap_y1 : float
    kick0 : str
    L : float
    lord_status : str
    matrix : str
    mode_flip0 : bool
    mode_flip1 : bool
    offset_moves_aperture : bool
    p0c : float
    px0 : float
    px1 : float
    py0 : float
    py1 : float
    pz0 : float
    pz1 : float
    recalc : bool
    ref_time_start : float
    slave_status : str
    units : dict[str, str]
        Per-attribute unit information.
    x0 : float
    x1 : float
    x1_limit : float
    x2_limit : float
    y0 : float
    y1 : float
    y1_limit : float
    y2_limit : float
    z0 : float
    z1 : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Match"]  # = "Match"

    alpha_a0: float = ROField(default=0.0, attr="alpha_a0", tao_name="ALPHA_A0")
    alpha_a1: float = ROField(default=0.0, attr="alpha_a1", tao_name="ALPHA_A1")
    alpha_b0: float = ROField(default=0.0, attr="alpha_b0", tao_name="ALPHA_B0")
    alpha_b1: float = ROField(default=0.0, attr="alpha_b1", tao_name="ALPHA_B1")
    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    beta_a0: float = ROField(default=0.0, attr="beta_a0", tao_name="BETA_A0")
    beta_a1: float = ROField(default=0.0, attr="beta_a1", tao_name="BETA_A1")
    beta_b0: float = ROField(default=0.0, attr="beta_b0", tao_name="BETA_B0")
    beta_b1: float = ROField(default=0.0, attr="beta_b1", tao_name="BETA_B1")
    c11_mat0: float = ROField(default=0.0, attr="c11_mat0", tao_name="C11_MAT0")
    c11_mat1: float = ROField(default=0.0, attr="c11_mat1", tao_name="C11_MAT1")
    c12_mat0: float = ROField(default=0.0, attr="c12_mat0", tao_name="C12_MAT0")
    c12_mat1: float = ROField(default=0.0, attr="c12_mat1", tao_name="C12_MAT1")
    c21_mat0: float = ROField(default=0.0, attr="c21_mat0", tao_name="C21_MAT0")
    c21_mat1: float = ROField(default=0.0, attr="c21_mat1", tao_name="C21_MAT1")
    c22_mat0: float = ROField(default=0.0, attr="c22_mat0", tao_name="C22_MAT0")
    c22_mat1: float = ROField(default=0.0, attr="c22_mat1", tao_name="C22_MAT1")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    delta_time: float = ROField(default=0.0, attr="delta_time", tao_name="DELTA_TIME")
    dphi_a: float = ROField(default=0.0, attr="dphi_a", tao_name="DPHI_A")
    dphi_b: float = ROField(default=0.0, attr="dphi_b", tao_name="DPHI_B")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    eta_x0: float = ROField(default=0.0, attr="eta_x0", tao_name="ETA_X0")
    eta_x1: float = ROField(default=0.0, attr="eta_x1", tao_name="ETA_X1")
    eta_y0: float = ROField(default=0.0, attr="eta_y0", tao_name="ETA_Y0")
    eta_y1: float = ROField(default=0.0, attr="eta_y1", tao_name="ETA_Y1")
    etap_x0: float = ROField(default=0.0, attr="etap_x0", tao_name="ETAP_X0")
    etap_x1: float = ROField(default=0.0, attr="etap_x1", tao_name="ETAP_X1")
    etap_y0: float = ROField(default=0.0, attr="etap_y0", tao_name="ETAP_Y0")
    etap_y1: float = ROField(default=0.0, attr="etap_y1", tao_name="ETAP_Y1")
    kick0: str = ROField(default="", attr="kick0", tao_name="KICK0")
    L: float = ROField(default=0.0)
    lord_status: str = ROField(default="")
    matrix: str = ROField(default="", attr="matrix", tao_name="MATRIX")
    mode_flip0: bool = ROField(default=False, attr="mode_flip0", tao_name="MODE_FLIP0")
    mode_flip1: bool = ROField(default=False, attr="mode_flip1", tao_name="MODE_FLIP1")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    px0: float = ROField(default=0.0, attr="px0", tao_name="PX0")
    px1: float = ROField(default=0.0, attr="px1", tao_name="PX1")
    py0: float = ROField(default=0.0, attr="py0", tao_name="PY0")
    py1: float = ROField(default=0.0, attr="py1", tao_name="PY1")
    pz0: float = ROField(default=0.0, attr="pz0", tao_name="PZ0")
    pz1: float = ROField(default=0.0, attr="pz1", tao_name="PZ1")
    recalc: bool = ROField(default=False, attr="recalc", tao_name="RECALC")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    slave_status: str = ROField(default="")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    x0: float = ROField(default=0.0, attr="x0", tao_name="X0")
    x1: float = ROField(default=0.0, attr="x1", tao_name="X1")
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    y0: float = ROField(default=0.0, attr="y0", tao_name="Y0")
    y1: float = ROField(default=0.0, attr="y1", tao_name="Y1")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    z0: float = ROField(default=0.0, attr="z0", tao_name="Z0")
    z1: float = ROField(default=0.0, attr="z1", tao_name="Z1")


class MonitorAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 20`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    bl_hkick : float
    bl_vkick : float
    crunch : float
    crunch_calib : float
    csr_ds_step : float
    de_eta_meas : float
    delta_ref_time : float
    ds_step : float
    e_tot : float
    fringe_at : str
    fringe_type : str
    hkick : float
    integrator_order : int
    L : float
    lord_pad1 : float
    lord_pad2 : float
    lord_status : str
    n_sample : float
    noise : float
    num_steps : int
    offset_moves_aperture : bool
    osc_amplitude : float
    p0c : float
    ref_time_start : float
    slave_status : str
    spin_fringe_on : bool
    static_linear_map : bool
    tilt : float
    tilt_calib : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    vkick : float
    x1_limit : float
    x2_limit : float
    x_dispersion_calib : float
    x_dispersion_err : float
    x_gain_calib : float
    x_gain_err : float
    x_offset : float
    x_offset_calib : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_dispersion_calib : float
    y_dispersion_err : float
    y_gain_calib : float
    y_gain_err : float
    y_offset : float
    y_offset_calib : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Monitor"]  # = "Monitor"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    bl_hkick: float = ROField(default=0.0, attr="bl_hkick", tao_name="BL_HKICK")
    bl_vkick: float = ROField(default=0.0, attr="bl_vkick", tao_name="BL_VKICK")
    crunch: float = ROField(default=0.0, attr="crunch", tao_name="CRUNCH")
    crunch_calib: float = ROField(default=0.0, attr="crunch_calib", tao_name="CRUNCH_CALIB")
    csr_ds_step: float = ROField(default=0.0, attr="csr_ds_step", tao_name="CSR_DS_STEP")
    de_eta_meas: float = ROField(default=0.0, attr="de_eta_meas", tao_name="DE_ETA_MEAS")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    ds_step: float = ROField(default=0.0, attr="ds_step", tao_name="DS_STEP")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    fringe_at: str = ROField(default="", attr="fringe_at", tao_name="FRINGE_AT")
    fringe_type: str = ROField(default="", attr="fringe_type", tao_name="FRINGE_TYPE")
    hkick: float = ROField(default=0.0, attr="hkick", tao_name="HKICK")
    integrator_order: int = ROField(default=0, attr="integrator_order", tao_name="INTEGRATOR_ORDER")
    L: float = ROField(default=0.0)
    lord_pad1: float = ROField(default=0.0, attr="lord_pad1", tao_name="LORD_PAD1")
    lord_pad2: float = ROField(default=0.0, attr="lord_pad2", tao_name="LORD_PAD2")
    lord_status: str = ROField(default="")
    n_sample: float = ROField(default=0.0, attr="n_sample", tao_name="N_SAMPLE")
    noise: float = ROField(default=0.0, attr="noise", tao_name="NOISE")
    num_steps: int = ROField(default=0, attr="num_steps", tao_name="NUM_STEPS")
    offset_moves_aperture: bool = Field(default=False)
    osc_amplitude: float = ROField(default=0.0, attr="osc_amplitude", tao_name="OSC_AMPLITUDE")
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    slave_status: str = ROField(default="")
    spin_fringe_on: bool = ROField(default=False, attr="spin_fringe_on", tao_name="SPIN_FRINGE_ON")
    static_linear_map: bool = ROField(
        default=False, attr="static_linear_map", tao_name="STATIC_LINEAR_MAP"
    )
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_calib: float = ROField(default=0.0, attr="tilt_calib", tao_name="TILT_CALIB")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    vkick: float = ROField(default=0.0, attr="vkick", tao_name="VKICK")
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_dispersion_calib: float = ROField(
        default=0.0, attr="x_dispersion_calib", tao_name="X_DISPERSION_CALIB"
    )
    x_dispersion_err: float = ROField(
        default=0.0, attr="x_dispersion_err", tao_name="X_DISPERSION_ERR"
    )
    x_gain_calib: float = ROField(default=0.0, attr="x_gain_calib", tao_name="X_GAIN_CALIB")
    x_gain_err: float = ROField(default=0.0, attr="x_gain_err", tao_name="X_GAIN_ERR")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_calib: float = ROField(default=0.0, attr="x_offset_calib", tao_name="X_OFFSET_CALIB")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_dispersion_calib: float = ROField(
        default=0.0, attr="y_dispersion_calib", tao_name="Y_DISPERSION_CALIB"
    )
    y_dispersion_err: float = ROField(
        default=0.0, attr="y_dispersion_err", tao_name="Y_DISPERSION_ERR"
    )
    y_gain_calib: float = ROField(default=0.0, attr="y_gain_calib", tao_name="Y_GAIN_CALIB")
    y_gain_err: float = ROField(default=0.0, attr="y_gain_err", tao_name="Y_GAIN_ERR")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_calib: float = ROField(default=0.0, attr="y_offset_calib", tao_name="Y_OFFSET_CALIB")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class Multilayer_MirrorAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 2`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    d1_thickness : float
    d2_thickness : float
    delta_ref_time : float
    e_tot : float
    graze_angle : float
    L : float
    lord_status : str
    n_cell : int
    offset_moves_aperture : bool
    p0c : float
    ref_tilt : float
    ref_tilt_tot : float
    ref_time_start : float
    ref_wavelength : float
    slave_status : str
    tilt : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    v1_unitcell : float
    v2_unitcell : float
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Multilayer_Mirror"]  # = "Multilayer_Mirror"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    d1_thickness: float = ROField(default=0.0, attr="d1_thickness", tao_name="D1_THICKNESS")
    d2_thickness: float = ROField(default=0.0, attr="d2_thickness", tao_name="D2_THICKNESS")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    graze_angle: float = ROField(default=0.0, attr="graze_angle", tao_name="GRAZE_ANGLE")
    L: float = ROField(default=0.0)
    lord_status: str = ROField(default="")
    n_cell: int = ROField(default=0, attr="n_cell", tao_name="N_CELL")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    ref_tilt: float = ROField(default=0.0, attr="ref_tilt", tao_name="REF_TILT")
    ref_tilt_tot: float = ROField(default=0.0, attr="ref_tilt_tot", tao_name="REF_TILT_TOT")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    ref_wavelength: float = ROField(default=0.0, attr="ref_wavelength", tao_name="REF_WAVELENGTH")
    slave_status: str = ROField(default="")
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    v1_unitcell: float = ROField(default=0.0, attr="v1_unitcell", tao_name="V1_UNITCELL")
    v2_unitcell: float = ROField(default=0.0, attr="v2_unitcell", tao_name="V2_UNITCELL")
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class MultipoleAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 21`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    delta_ref_time : float
    e_tot : float
    field_master : bool
    L : float
    lord_status : str
    offset_moves_aperture : bool
    p0c : float
    ref_time_start : float
    slave_status : str
    tilt : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_offset_tot : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_offset_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Multipole"]  # = "Multipole"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    field_master: bool = Field(default=False)
    L: float = ROField(default=0.0)
    lord_status: str = ROField(default="")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    slave_status: str = ROField(default="")
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class OctupoleAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 22`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    b3_gradient : float
    bl_hkick : float
    bl_vkick : float
    csr_ds_step : float
    delta_ref_time : float
    ds_step : float
    e_tot : float
    field_master : bool
    fringe_at : str
    fringe_type : str
    hkick : float
    integrator_order : int
    k3 : float
    L : float
    lord_pad1 : float
    lord_pad2 : float
    lord_status : str
    num_steps : int
    offset_moves_aperture : bool
    p0c : float
    ptc_canonical_coords : bool
    r0_elec : float
    r0_mag : float
    ref_time_start : float
    slave_status : str
    spin_fringe_on : bool
    static_linear_map : bool
    tilt : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    vkick : float
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Octupole"]  # = "Octupole"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    b3_gradient: float = ROField(default=0.0, attr="b3_gradient", tao_name="B3_GRADIENT")
    bl_hkick: float = ROField(default=0.0, attr="bl_hkick", tao_name="BL_HKICK")
    bl_vkick: float = ROField(default=0.0, attr="bl_vkick", tao_name="BL_VKICK")
    csr_ds_step: float = ROField(default=0.0, attr="csr_ds_step", tao_name="CSR_DS_STEP")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    ds_step: float = ROField(default=0.0, attr="ds_step", tao_name="DS_STEP")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    field_master: bool = Field(default=False)
    fringe_at: str = ROField(default="", attr="fringe_at", tao_name="FRINGE_AT")
    fringe_type: str = ROField(default="", attr="fringe_type", tao_name="FRINGE_TYPE")
    hkick: float = ROField(default=0.0, attr="hkick", tao_name="HKICK")
    integrator_order: int = ROField(default=0, attr="integrator_order", tao_name="INTEGRATOR_ORDER")
    k3: float = ROField(default=0.0, attr="k3", tao_name="K3")
    L: float = ROField(default=0.0)
    lord_pad1: float = ROField(default=0.0, attr="lord_pad1", tao_name="LORD_PAD1")
    lord_pad2: float = ROField(default=0.0, attr="lord_pad2", tao_name="LORD_PAD2")
    lord_status: str = ROField(default="")
    num_steps: int = ROField(default=0, attr="num_steps", tao_name="NUM_STEPS")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    ptc_canonical_coords: bool = ROField(
        default=False, attr="ptc_canonical_coords", tao_name="PTC_CANONICAL_COORDS"
    )
    r0_elec: float = ROField(default=0.0, attr="r0_elec", tao_name="R0_ELEC")
    r0_mag: float = ROField(default=0.0, attr="r0_mag", tao_name="R0_MAG")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    slave_status: str = ROField(default="")
    spin_fringe_on: bool = ROField(default=False, attr="spin_fringe_on", tao_name="SPIN_FRINGE_ON")
    static_linear_map: bool = ROField(
        default=False, attr="static_linear_map", tao_name="STATIC_LINEAR_MAP"
    )
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    vkick: float = ROField(default=0.0, attr="vkick", tao_name="VKICK")
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class OverlayAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 14`, for example.

    Attributes
    ----------
    gang : bool
    interpolation : str
    lord_status : str
    slave_status : str
    units : dict[str, str]
        Per-attribute unit information.
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Overlay"]  # = "Overlay"

    gang: bool = ROField(default=False, attr="gang", tao_name="GANG")
    interpolation: str = ROField(default="", attr="interpolation", tao_name="INTERPOLATION")
    lord_status: str = ROField(default="")
    slave_status: str = ROField(default="")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )


class PatchAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 23`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    csr_ds_step : float
    delta_ref_time : float
    downstream_ele_dir : int
    e_tot : float
    e_tot_offset : float
    e_tot_set : float
    e_tot_start : float
    flexible : bool
    L : float
    lord_status : str
    offset_moves_aperture : bool
    p0c : float
    p0c_set : float
    p0c_start : float
    ref_coords : str
    ref_time_start : float
    slave_status : str
    t_offset : float
    tilt : float
    units : dict[str, str]
        Per-attribute unit information.
    upstream_ele_dir : int
    user_sets_length : bool
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_pitch : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_pitch : float
    z_offset : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Patch"]  # = "Patch"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    csr_ds_step: float = ROField(default=0.0, attr="csr_ds_step", tao_name="CSR_DS_STEP")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    downstream_ele_dir: int = ROField(
        default=0, attr="downstream_ele_dir", tao_name="DOWNSTREAM_ELE_DIR"
    )
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    e_tot_offset: float = ROField(default=0.0, attr="e_tot_offset", tao_name="E_TOT_OFFSET")
    e_tot_set: float = ROField(default=0.0, attr="e_tot_set", tao_name="E_TOT_SET")
    e_tot_start: float = ROField(default=0.0, attr="e_tot_start", tao_name="E_TOT_START")
    flexible: bool = ROField(default=False, attr="flexible", tao_name="FLEXIBLE")
    L: float = ROField(default=0.0)
    lord_status: str = ROField(default="")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    p0c_set: float = ROField(default=0.0, attr="p0c_set", tao_name="P0C_SET")
    p0c_start: float = ROField(default=0.0, attr="p0c_start", tao_name="P0C_START")
    ref_coords: str = ROField(default="", attr="ref_coords", tao_name="REF_COORDS")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    slave_status: str = ROField(default="")
    t_offset: float = ROField(default=0.0, attr="t_offset", tao_name="T_OFFSET")
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    upstream_ele_dir: int = ROField(default=0, attr="upstream_ele_dir", tao_name="UPSTREAM_ELE_DIR")
    user_sets_length: bool = ROField(
        default=False, attr="user_sets_length", tao_name="USER_SETS_LENGTH"
    )
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")


class Photon_ForkAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 2`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    delta_ref_time : float
    direction : int
    e_tot : float
    ix_to_branch : int
    ix_to_element : int
    L : float
    lord_status : str
    new_branch : bool
    offset_moves_aperture : bool
    p0c : float
    ref_time_start : float
    slave_status : str
    units : dict[str, str]
        Per-attribute unit information.
    x1_limit : float
    x2_limit : float
    y1_limit : float
    y2_limit : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Photon_Fork"]  # = "Photon_Fork"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    direction: int = ROField(default=0, attr="direction", tao_name="DIRECTION")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    ix_to_branch: int = ROField(default=0, attr="ix_to_branch", tao_name="IX_TO_BRANCH")
    ix_to_element: int = ROField(default=0, attr="ix_to_element", tao_name="IX_TO_ELEMENT")
    L: float = ROField(default=0.0)
    lord_status: str = ROField(default="")
    new_branch: bool = ROField(default=False, attr="new_branch", tao_name="NEW_BRANCH")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    slave_status: str = ROField(default="")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")


class Photon_InitAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 1`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    delta_ref_time : float
    ds_slice : float
    e2_center : float
    e2_probability : float
    e_center : float
    e_center_relative_to_ref : bool
    e_field_x : float
    e_field_y : float
    e_tot : float
    energy_distribution : str
    L : float
    lord_status : str
    offset_moves_aperture : bool
    p0c : float
    ref_time_start : float
    ref_wavelength : float
    scale_field_to_one : bool
    sig_e : float
    sig_e2 : float
    sig_vx : float
    sig_vy : float
    sig_x : float
    sig_y : float
    sig_z : float
    slave_status : str
    spatial_distribution : str
    tilt : float
    tilt_tot : float
    transverse_sigma_cut : float
    units : dict[str, str]
        Per-attribute unit information.
    velocity_distribution : str
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Photon_Init"]  # = "Photon_Init"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    ds_slice: float = ROField(default=0.0, attr="ds_slice", tao_name="DS_SLICE")
    e2_center: float = ROField(default=0.0, attr="e2_center", tao_name="E2_CENTER")
    e2_probability: float = ROField(default=0.0, attr="e2_probability", tao_name="E2_PROBABILITY")
    e_center: float = ROField(default=0.0, attr="e_center", tao_name="E_CENTER")
    e_center_relative_to_ref: bool = ROField(
        default=False, attr="e_center_relative_to_ref", tao_name="E_CENTER_RELATIVE_TO_REF"
    )
    e_field_x: float = ROField(default=0.0, attr="e_field_x", tao_name="E_FIELD_X")
    e_field_y: float = ROField(default=0.0, attr="e_field_y", tao_name="E_FIELD_Y")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    energy_distribution: str = ROField(
        default="", attr="energy_distribution", tao_name="ENERGY_DISTRIBUTION"
    )
    L: float = ROField(default=0.0)
    lord_status: str = ROField(default="")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    ref_wavelength: float = ROField(default=0.0, attr="ref_wavelength", tao_name="REF_WAVELENGTH")
    scale_field_to_one: bool = ROField(
        default=False, attr="scale_field_to_one", tao_name="SCALE_FIELD_TO_ONE"
    )
    sig_e: float = ROField(default=0.0, attr="sig_e", tao_name="SIG_E")
    sig_e2: float = ROField(default=0.0, attr="sig_e2", tao_name="SIG_E2")
    sig_vx: float = ROField(default=0.0, attr="sig_vx", tao_name="SIG_VX")
    sig_vy: float = ROField(default=0.0, attr="sig_vy", tao_name="SIG_VY")
    sig_x: float = ROField(default=0.0, attr="sig_x", tao_name="SIG_X")
    sig_y: float = ROField(default=0.0, attr="sig_y", tao_name="SIG_Y")
    sig_z: float = ROField(default=0.0, attr="sig_z", tao_name="SIG_Z")
    slave_status: str = ROField(default="")
    spatial_distribution: str = ROField(
        default="", attr="spatial_distribution", tao_name="SPATIAL_DISTRIBUTION"
    )
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    transverse_sigma_cut: float = ROField(
        default=0.0, attr="transverse_sigma_cut", tao_name="TRANSVERSE_SIGMA_CUT"
    )
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    velocity_distribution: str = ROField(
        default="", attr="velocity_distribution", tao_name="VELOCITY_DISTRIBUTION"
    )
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class PipeAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 2`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    bl_hkick : float
    bl_vkick : float
    crunch : float
    crunch_calib : float
    csr_ds_step : float
    de_eta_meas : float
    delta_ref_time : float
    ds_step : float
    e_tot : float
    fringe_at : str
    fringe_type : str
    hkick : float
    integrator_order : int
    L : float
    lord_pad1 : float
    lord_pad2 : float
    lord_status : str
    n_sample : float
    noise : float
    num_steps : int
    offset_moves_aperture : bool
    osc_amplitude : float
    p0c : float
    ref_time_start : float
    slave_status : str
    spin_fringe_on : bool
    static_linear_map : bool
    tilt : float
    tilt_calib : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    vkick : float
    x1_limit : float
    x2_limit : float
    x_dispersion_calib : float
    x_dispersion_err : float
    x_gain_calib : float
    x_gain_err : float
    x_offset : float
    x_offset_calib : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_dispersion_calib : float
    y_dispersion_err : float
    y_gain_calib : float
    y_gain_err : float
    y_offset : float
    y_offset_calib : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Pipe"]  # = "Pipe"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    bl_hkick: float = ROField(default=0.0, attr="bl_hkick", tao_name="BL_HKICK")
    bl_vkick: float = ROField(default=0.0, attr="bl_vkick", tao_name="BL_VKICK")
    crunch: float = ROField(default=0.0, attr="crunch", tao_name="CRUNCH")
    crunch_calib: float = ROField(default=0.0, attr="crunch_calib", tao_name="CRUNCH_CALIB")
    csr_ds_step: float = ROField(default=0.0, attr="csr_ds_step", tao_name="CSR_DS_STEP")
    de_eta_meas: float = ROField(default=0.0, attr="de_eta_meas", tao_name="DE_ETA_MEAS")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    ds_step: float = ROField(default=0.0, attr="ds_step", tao_name="DS_STEP")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    fringe_at: str = ROField(default="", attr="fringe_at", tao_name="FRINGE_AT")
    fringe_type: str = ROField(default="", attr="fringe_type", tao_name="FRINGE_TYPE")
    hkick: float = ROField(default=0.0, attr="hkick", tao_name="HKICK")
    integrator_order: int = ROField(default=0, attr="integrator_order", tao_name="INTEGRATOR_ORDER")
    L: float = ROField(default=0.0)
    lord_pad1: float = ROField(default=0.0, attr="lord_pad1", tao_name="LORD_PAD1")
    lord_pad2: float = ROField(default=0.0, attr="lord_pad2", tao_name="LORD_PAD2")
    lord_status: str = ROField(default="")
    n_sample: float = ROField(default=0.0, attr="n_sample", tao_name="N_SAMPLE")
    noise: float = ROField(default=0.0, attr="noise", tao_name="NOISE")
    num_steps: int = ROField(default=0, attr="num_steps", tao_name="NUM_STEPS")
    offset_moves_aperture: bool = Field(default=False)
    osc_amplitude: float = ROField(default=0.0, attr="osc_amplitude", tao_name="OSC_AMPLITUDE")
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    slave_status: str = ROField(default="")
    spin_fringe_on: bool = ROField(default=False, attr="spin_fringe_on", tao_name="SPIN_FRINGE_ON")
    static_linear_map: bool = ROField(
        default=False, attr="static_linear_map", tao_name="STATIC_LINEAR_MAP"
    )
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_calib: float = ROField(default=0.0, attr="tilt_calib", tao_name="TILT_CALIB")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    vkick: float = ROField(default=0.0, attr="vkick", tao_name="VKICK")
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_dispersion_calib: float = ROField(
        default=0.0, attr="x_dispersion_calib", tao_name="X_DISPERSION_CALIB"
    )
    x_dispersion_err: float = ROField(
        default=0.0, attr="x_dispersion_err", tao_name="X_DISPERSION_ERR"
    )
    x_gain_calib: float = ROField(default=0.0, attr="x_gain_calib", tao_name="X_GAIN_CALIB")
    x_gain_err: float = ROField(default=0.0, attr="x_gain_err", tao_name="X_GAIN_ERR")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_calib: float = ROField(default=0.0, attr="x_offset_calib", tao_name="X_OFFSET_CALIB")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_dispersion_calib: float = ROField(
        default=0.0, attr="y_dispersion_calib", tao_name="Y_DISPERSION_CALIB"
    )
    y_dispersion_err: float = ROField(
        default=0.0, attr="y_dispersion_err", tao_name="Y_DISPERSION_ERR"
    )
    y_gain_calib: float = ROField(default=0.0, attr="y_gain_calib", tao_name="Y_GAIN_CALIB")
    y_gain_err: float = ROField(default=0.0, attr="y_gain_err", tao_name="Y_GAIN_ERR")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_calib: float = ROField(default=0.0, attr="y_offset_calib", tao_name="Y_OFFSET_CALIB")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class QuadrupoleAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 24`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    b1_gradient : float
    bl_hkick : float
    bl_vkick : float
    csr_ds_step : float
    delta_ref_time : float
    ds_step : float
    e_tot : float
    field_master : bool
    fq1 : float
    fq2 : float
    fringe_at : str
    fringe_type : str
    hkick : float
    integrator_order : int
    k1 : float
    L : float
    lord_pad1 : float
    lord_pad2 : float
    lord_status : str
    num_steps : int
    offset_moves_aperture : bool
    p0c : float
    ptc_canonical_coords : bool
    r0_elec : float
    r0_mag : float
    ref_time_start : float
    slave_status : str
    spin_fringe_on : bool
    static_linear_map : bool
    tilt : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    vkick : float
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Quadrupole"]  # = "Quadrupole"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    b1_gradient: float = ROField(default=0.0, attr="b1_gradient", tao_name="B1_GRADIENT")
    bl_hkick: float = ROField(default=0.0, attr="bl_hkick", tao_name="BL_HKICK")
    bl_vkick: float = ROField(default=0.0, attr="bl_vkick", tao_name="BL_VKICK")
    csr_ds_step: float = ROField(default=0.0, attr="csr_ds_step", tao_name="CSR_DS_STEP")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    ds_step: float = ROField(default=0.0, attr="ds_step", tao_name="DS_STEP")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    field_master: bool = Field(default=False)
    fq1: float = ROField(default=0.0, attr="fq1", tao_name="FQ1")
    fq2: float = ROField(default=0.0, attr="fq2", tao_name="FQ2")
    fringe_at: str = ROField(default="", attr="fringe_at", tao_name="FRINGE_AT")
    fringe_type: str = ROField(default="", attr="fringe_type", tao_name="FRINGE_TYPE")
    hkick: float = ROField(default=0.0, attr="hkick", tao_name="HKICK")
    integrator_order: int = ROField(default=0, attr="integrator_order", tao_name="INTEGRATOR_ORDER")
    k1: float = ROField(default=0.0, attr="k1", tao_name="K1")
    L: float = ROField(default=0.0)
    lord_pad1: float = ROField(default=0.0, attr="lord_pad1", tao_name="LORD_PAD1")
    lord_pad2: float = ROField(default=0.0, attr="lord_pad2", tao_name="LORD_PAD2")
    lord_status: str = ROField(default="")
    num_steps: int = ROField(default=0, attr="num_steps", tao_name="NUM_STEPS")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    ptc_canonical_coords: bool = ROField(
        default=False, attr="ptc_canonical_coords", tao_name="PTC_CANONICAL_COORDS"
    )
    r0_elec: float = ROField(default=0.0, attr="r0_elec", tao_name="R0_ELEC")
    r0_mag: float = ROField(default=0.0, attr="r0_mag", tao_name="R0_MAG")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    slave_status: str = ROField(default="")
    spin_fringe_on: bool = ROField(default=False, attr="spin_fringe_on", tao_name="SPIN_FRINGE_ON")
    static_linear_map: bool = ROField(
        default=False, attr="static_linear_map", tao_name="STATIC_LINEAR_MAP"
    )
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    vkick: float = ROField(default=0.0, attr="vkick", tao_name="VKICK")
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class RamperAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 26`, for example.

    Attributes
    ----------
    interpolation : str
    lord_status : str
    slave_status : str
    units : dict[str, str]
        Per-attribute unit information.
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Ramper"]  # = "Ramper"

    interpolation: str = ROField(default="", attr="interpolation", tao_name="INTERPOLATION")
    lord_status: str = ROField(default="")
    slave_status: str = ROField(default="")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )


class RCollimatorAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 29`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    bl_hkick : float
    bl_vkick : float
    csr_ds_step : float
    delta_ref_time : float
    ds_step : float
    e_tot : float
    fringe_at : str
    fringe_type : str
    hkick : float
    integrator_order : int
    L : float
    lord_pad1 : float
    lord_pad2 : float
    lord_status : str
    num_steps : int
    offset_moves_aperture : bool
    p0c : float
    px_aperture_center : float
    px_aperture_width2 : float
    py_aperture_center : float
    py_aperture_width2 : float
    pz_aperture_center : float
    pz_aperture_width2 : float
    ref_time_start : float
    slave_status : str
    spin_fringe_on : bool
    static_linear_map : bool
    tilt : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    vkick : float
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_aperture_center : float
    z_aperture_width2 : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["RCollimator"]  # = "RCollimator"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    bl_hkick: float = ROField(default=0.0, attr="bl_hkick", tao_name="BL_HKICK")
    bl_vkick: float = ROField(default=0.0, attr="bl_vkick", tao_name="BL_VKICK")
    csr_ds_step: float = ROField(default=0.0, attr="csr_ds_step", tao_name="CSR_DS_STEP")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    ds_step: float = ROField(default=0.0, attr="ds_step", tao_name="DS_STEP")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    fringe_at: str = ROField(default="", attr="fringe_at", tao_name="FRINGE_AT")
    fringe_type: str = ROField(default="", attr="fringe_type", tao_name="FRINGE_TYPE")
    hkick: float = ROField(default=0.0, attr="hkick", tao_name="HKICK")
    integrator_order: int = ROField(default=0, attr="integrator_order", tao_name="INTEGRATOR_ORDER")
    L: float = ROField(default=0.0)
    lord_pad1: float = ROField(default=0.0, attr="lord_pad1", tao_name="LORD_PAD1")
    lord_pad2: float = ROField(default=0.0, attr="lord_pad2", tao_name="LORD_PAD2")
    lord_status: str = ROField(default="")
    num_steps: int = ROField(default=0, attr="num_steps", tao_name="NUM_STEPS")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    px_aperture_center: float = ROField(
        default=0.0, attr="px_aperture_center", tao_name="PX_APERTURE_CENTER"
    )
    px_aperture_width2: float = ROField(
        default=0.0, attr="px_aperture_width2", tao_name="PX_APERTURE_WIDTH2"
    )
    py_aperture_center: float = ROField(
        default=0.0, attr="py_aperture_center", tao_name="PY_APERTURE_CENTER"
    )
    py_aperture_width2: float = ROField(
        default=0.0, attr="py_aperture_width2", tao_name="PY_APERTURE_WIDTH2"
    )
    pz_aperture_center: float = ROField(
        default=0.0, attr="pz_aperture_center", tao_name="PZ_APERTURE_CENTER"
    )
    pz_aperture_width2: float = ROField(
        default=0.0, attr="pz_aperture_width2", tao_name="PZ_APERTURE_WIDTH2"
    )
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    slave_status: str = ROField(default="")
    spin_fringe_on: bool = ROField(default=False, attr="spin_fringe_on", tao_name="SPIN_FRINGE_ON")
    static_linear_map: bool = ROField(
        default=False, attr="static_linear_map", tao_name="STATIC_LINEAR_MAP"
    )
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    vkick: float = ROField(default=0.0, attr="vkick", tao_name="VKICK")
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_aperture_center: float = ROField(
        default=0.0, attr="z_aperture_center", tao_name="Z_APERTURE_CENTER"
    )
    z_aperture_width2: float = ROField(
        default=0.0, attr="z_aperture_width2", tao_name="Z_APERTURE_WIDTH2"
    )
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class RFCavityAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 30`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    autoscale_amplitude : bool
    autoscale_phase : bool
    bl_hkick : float
    bl_vkick : float
    cavity_type : str
    coupler_angle : float
    coupler_at : str
    coupler_phase : float
    coupler_strength : float
    csr_ds_step : float
    delta_ref_time : float
    ds_step : float
    e_tot : float
    field_autoscale : float
    fringe_at : str
    fringe_type : str
    gradient : float
    harmon : float
    harmon_master : bool
    hkick : float
    integrator_order : int
    L : float
    l_active : float
    longitudinal_mode : int
    lord_pad1 : float
    lord_pad2 : float
    lord_status : str
    n_cell : int
    num_steps : int
    offset_moves_aperture : bool
    p0c : float
    phi0 : float
    phi0_autoscale : float
    phi0_multipass : float
    ref_time_start : float
    rf_frequency : float
    rf_wavelength : float
    slave_status : str
    spin_fringe_on : bool
    static_linear_map : bool
    tilt : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    vkick : float
    voltage : float
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["RFCavity"]  # = "RFCavity"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    autoscale_amplitude: bool = ROField(
        default=False, attr="autoscale_amplitude", tao_name="AUTOSCALE_AMPLITUDE"
    )
    autoscale_phase: bool = ROField(
        default=False, attr="autoscale_phase", tao_name="AUTOSCALE_PHASE"
    )
    bl_hkick: float = ROField(default=0.0, attr="bl_hkick", tao_name="BL_HKICK")
    bl_vkick: float = ROField(default=0.0, attr="bl_vkick", tao_name="BL_VKICK")
    cavity_type: str = ROField(default="", attr="cavity_type", tao_name="CAVITY_TYPE")
    coupler_angle: float = ROField(default=0.0, attr="coupler_angle", tao_name="COUPLER_ANGLE")
    coupler_at: str = ROField(default="", attr="coupler_at", tao_name="COUPLER_AT")
    coupler_phase: float = ROField(default=0.0, attr="coupler_phase", tao_name="COUPLER_PHASE")
    coupler_strength: float = ROField(
        default=0.0, attr="coupler_strength", tao_name="COUPLER_STRENGTH"
    )
    csr_ds_step: float = ROField(default=0.0, attr="csr_ds_step", tao_name="CSR_DS_STEP")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    ds_step: float = ROField(default=0.0, attr="ds_step", tao_name="DS_STEP")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    field_autoscale: float = ROField(
        default=0.0, attr="field_autoscale", tao_name="FIELD_AUTOSCALE"
    )
    fringe_at: str = ROField(default="", attr="fringe_at", tao_name="FRINGE_AT")
    fringe_type: str = ROField(default="", attr="fringe_type", tao_name="FRINGE_TYPE")
    gradient: float = ROField(default=0.0, attr="gradient", tao_name="GRADIENT")
    harmon: float = ROField(default=0.0, attr="harmon", tao_name="HARMON")
    harmon_master: bool = ROField(default=False, attr="harmon_master", tao_name="HARMON_MASTER")
    hkick: float = ROField(default=0.0, attr="hkick", tao_name="HKICK")
    integrator_order: int = ROField(default=0, attr="integrator_order", tao_name="INTEGRATOR_ORDER")
    L: float = ROField(default=0.0)
    l_active: float = ROField(default=0.0, attr="l_active", tao_name="L_ACTIVE")
    longitudinal_mode: int = ROField(
        default=0, attr="longitudinal_mode", tao_name="LONGITUDINAL_MODE"
    )
    lord_pad1: float = ROField(default=0.0, attr="lord_pad1", tao_name="LORD_PAD1")
    lord_pad2: float = ROField(default=0.0, attr="lord_pad2", tao_name="LORD_PAD2")
    lord_status: str = ROField(default="")
    n_cell: int = ROField(default=0, attr="n_cell", tao_name="N_CELL")
    num_steps: int = ROField(default=0, attr="num_steps", tao_name="NUM_STEPS")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    phi0: float = ROField(default=0.0, attr="phi0", tao_name="PHI0")
    phi0_autoscale: float = ROField(default=0.0, attr="phi0_autoscale", tao_name="PHI0_AUTOSCALE")
    phi0_multipass: float = ROField(default=0.0, attr="phi0_multipass", tao_name="PHI0_MULTIPASS")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    rf_frequency: float = ROField(default=0.0, attr="rf_frequency", tao_name="RF_FREQUENCY")
    rf_wavelength: float = ROField(default=0.0, attr="rf_wavelength", tao_name="RF_WAVELENGTH")
    slave_status: str = ROField(default="")
    spin_fringe_on: bool = ROField(default=False, attr="spin_fringe_on", tao_name="SPIN_FRINGE_ON")
    static_linear_map: bool = ROField(
        default=False, attr="static_linear_map", tao_name="STATIC_LINEAR_MAP"
    )
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    vkick: float = ROField(default=0.0, attr="vkick", tao_name="VKICK")
    voltage: float = ROField(default=0.0, attr="voltage", tao_name="VOLTAGE")
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class Sad_MultAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 33`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    bs_field : float
    csr_ds_step : float
    delta_ref_time : float
    ds_step : float
    e1 : float
    e2 : float
    e_tot : float
    eps_step_scale : float
    fb1 : float
    fb2 : float
    fq1 : float
    fq2 : float
    fringe_at : str
    fringe_type : str
    integrator_order : int
    ks : float
    L : float
    lord_pad1 : float
    lord_pad2 : float
    lord_status : str
    num_steps : int
    offset_moves_aperture : bool
    p0c : float
    ref_time_start : float
    rho : float
    slave_status : str
    spin_fringe_on : bool
    static_linear_map : bool
    tilt : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_offset_mult : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_offset_mult : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Sad_Mult"]  # = "Sad_Mult"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    bs_field: float = ROField(default=0.0, attr="bs_field", tao_name="BS_FIELD")
    csr_ds_step: float = ROField(default=0.0, attr="csr_ds_step", tao_name="CSR_DS_STEP")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    ds_step: float = ROField(default=0.0, attr="ds_step", tao_name="DS_STEP")
    e1: float = ROField(default=0.0, attr="e1", tao_name="E1")
    e2: float = ROField(default=0.0, attr="e2", tao_name="E2")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    eps_step_scale: float = ROField(default=0.0, attr="eps_step_scale", tao_name="EPS_STEP_SCALE")
    fb1: float = ROField(default=0.0, attr="fb1", tao_name="FB1")
    fb2: float = ROField(default=0.0, attr="fb2", tao_name="FB2")
    fq1: float = ROField(default=0.0, attr="fq1", tao_name="FQ1")
    fq2: float = ROField(default=0.0, attr="fq2", tao_name="FQ2")
    fringe_at: str = ROField(default="", attr="fringe_at", tao_name="FRINGE_AT")
    fringe_type: str = ROField(default="", attr="fringe_type", tao_name="FRINGE_TYPE")
    integrator_order: int = ROField(default=0, attr="integrator_order", tao_name="INTEGRATOR_ORDER")
    ks: float = ROField(default=0.0, attr="ks", tao_name="KS")
    L: float = ROField(default=0.0)
    lord_pad1: float = ROField(default=0.0, attr="lord_pad1", tao_name="LORD_PAD1")
    lord_pad2: float = ROField(default=0.0, attr="lord_pad2", tao_name="LORD_PAD2")
    lord_status: str = ROField(default="")
    num_steps: int = ROField(default=0, attr="num_steps", tao_name="NUM_STEPS")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    rho: float = ROField(default=0.0, attr="rho", tao_name="RHO")
    slave_status: str = ROField(default="")
    spin_fringe_on: bool = ROField(default=False, attr="spin_fringe_on", tao_name="SPIN_FRINGE_ON")
    static_linear_map: bool = ROField(
        default=False, attr="static_linear_map", tao_name="STATIC_LINEAR_MAP"
    )
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_mult: float = ROField(default=0.0, attr="x_offset_mult", tao_name="X_OFFSET_MULT")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_mult: float = ROField(default=0.0, attr="y_offset_mult", tao_name="Y_OFFSET_MULT")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class SBendAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 35`, for example.

    Attributes
    ----------
    angle : float
    aperture_at : str
    aperture_type : str
    b1_gradient : float
    b2_gradient : float
    b_field : float
    b_field_tot : float
    bl_hkick : float
    bl_vkick : float
    csr_ds_step : float
    db_field : float
    delta_ref_time : float
    dg : float
    ds_step : float
    e1 : float
    e2 : float
    e_tot : float
    exact_multipoles : str
    fiducial_pt : str
    field_master : bool
    fint : float
    fintx : float
    fringe_at : str
    fringe_type : str
    g : float
    g_tot : float
    h1 : float
    h2 : float
    hgap : float
    hgapx : float
    hkick : float
    integrator_order : int
    k1 : float
    k2 : float
    L : float
    l_chord : float
    l_rectangle : float
    l_sagitta : float
    lord_pad1 : float
    lord_pad2 : float
    lord_status : str
    num_steps : int
    offset_moves_aperture : bool
    p0c : float
    ptc_canonical_coords : bool
    ptc_field_geometry : str
    ptc_fringe_geometry : str
    r0_elec : float
    r0_mag : float
    ref_tilt : float
    ref_tilt_tot : float
    ref_time_start : float
    rho : float
    roll : float
    roll_tot : float
    slave_status : str
    spin_fringe_on : bool
    static_linear_map : bool
    units : dict[str, str]
        Per-attribute unit information.
    vkick : float
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["SBend"]  # = "SBend"

    angle: float = ROField(default=0.0, attr="angle", tao_name="ANGLE")
    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    b1_gradient: float = ROField(default=0.0, attr="b1_gradient", tao_name="B1_GRADIENT")
    b2_gradient: float = ROField(default=0.0, attr="b2_gradient", tao_name="B2_GRADIENT")
    b_field: float = ROField(default=0.0, attr="b_field", tao_name="B_FIELD")
    b_field_tot: float = ROField(default=0.0, attr="b_field_tot", tao_name="B_FIELD_TOT")
    bl_hkick: float = ROField(default=0.0, attr="bl_hkick", tao_name="BL_HKICK")
    bl_vkick: float = ROField(default=0.0, attr="bl_vkick", tao_name="BL_VKICK")
    csr_ds_step: float = ROField(default=0.0, attr="csr_ds_step", tao_name="CSR_DS_STEP")
    db_field: float = ROField(default=0.0, attr="db_field", tao_name="DB_FIELD")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    dg: float = ROField(default=0.0, attr="dg", tao_name="DG")
    ds_step: float = ROField(default=0.0, attr="ds_step", tao_name="DS_STEP")
    e1: float = ROField(default=0.0, attr="e1", tao_name="E1")
    e2: float = ROField(default=0.0, attr="e2", tao_name="E2")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    exact_multipoles: str = ROField(
        default="", attr="exact_multipoles", tao_name="EXACT_MULTIPOLES"
    )
    fiducial_pt: str = ROField(default="", attr="fiducial_pt", tao_name="FIDUCIAL_PT")
    field_master: bool = Field(default=False)
    fint: float = ROField(default=0.0, attr="fint", tao_name="FINT")
    fintx: float = ROField(default=0.0, attr="fintx", tao_name="FINTX")
    fringe_at: str = ROField(default="", attr="fringe_at", tao_name="FRINGE_AT")
    fringe_type: str = ROField(default="", attr="fringe_type", tao_name="FRINGE_TYPE")
    g: float = ROField(default=0.0, attr="g", tao_name="G")
    g_tot: float = ROField(default=0.0, attr="g_tot", tao_name="G_TOT")
    h1: float = ROField(default=0.0, attr="h1", tao_name="H1")
    h2: float = ROField(default=0.0, attr="h2", tao_name="H2")
    hgap: float = ROField(default=0.0, attr="hgap", tao_name="HGAP")
    hgapx: float = ROField(default=0.0, attr="hgapx", tao_name="HGAPX")
    hkick: float = ROField(default=0.0, attr="hkick", tao_name="HKICK")
    integrator_order: int = ROField(default=0, attr="integrator_order", tao_name="INTEGRATOR_ORDER")
    k1: float = ROField(default=0.0, attr="k1", tao_name="K1")
    k2: float = ROField(default=0.0, attr="k2", tao_name="K2")
    L: float = ROField(default=0.0)
    l_chord: float = ROField(default=0.0, attr="l_chord", tao_name="L_CHORD")
    l_rectangle: float = ROField(default=0.0, attr="l_rectangle", tao_name="L_RECTANGLE")
    l_sagitta: float = ROField(default=0.0, attr="l_sagitta", tao_name="L_SAGITTA")
    lord_pad1: float = ROField(default=0.0, attr="lord_pad1", tao_name="LORD_PAD1")
    lord_pad2: float = ROField(default=0.0, attr="lord_pad2", tao_name="LORD_PAD2")
    lord_status: str = ROField(default="")
    num_steps: int = ROField(default=0, attr="num_steps", tao_name="NUM_STEPS")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    ptc_canonical_coords: bool = ROField(
        default=False, attr="ptc_canonical_coords", tao_name="PTC_CANONICAL_COORDS"
    )
    ptc_field_geometry: str = ROField(
        default="", attr="ptc_field_geometry", tao_name="PTC_FIELD_GEOMETRY"
    )
    ptc_fringe_geometry: str = ROField(
        default="", attr="ptc_fringe_geometry", tao_name="PTC_FRINGE_GEOMETRY"
    )
    r0_elec: float = ROField(default=0.0, attr="r0_elec", tao_name="R0_ELEC")
    r0_mag: float = ROField(default=0.0, attr="r0_mag", tao_name="R0_MAG")
    ref_tilt: float = ROField(default=0.0, attr="ref_tilt", tao_name="REF_TILT")
    ref_tilt_tot: float = ROField(default=0.0, attr="ref_tilt_tot", tao_name="REF_TILT_TOT")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    rho: float = ROField(default=0.0, attr="rho", tao_name="RHO")
    roll: float = ROField(default=0.0, attr="roll", tao_name="ROLL")
    roll_tot: float = ROField(default=0.0, attr="roll_tot", tao_name="ROLL_TOT")
    slave_status: str = ROField(default="")
    spin_fringe_on: bool = ROField(default=False, attr="spin_fringe_on", tao_name="SPIN_FRINGE_ON")
    static_linear_map: bool = ROField(
        default=False, attr="static_linear_map", tao_name="STATIC_LINEAR_MAP"
    )
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    vkick: float = ROField(default=0.0, attr="vkick", tao_name="VKICK")
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class SextupoleAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 42`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    b2_gradient : float
    bl_hkick : float
    bl_vkick : float
    csr_ds_step : float
    delta_ref_time : float
    ds_step : float
    e_tot : float
    field_master : bool
    fringe_at : str
    fringe_type : str
    hkick : float
    integrator_order : int
    k2 : float
    L : float
    lord_pad1 : float
    lord_pad2 : float
    lord_status : str
    num_steps : int
    offset_moves_aperture : bool
    p0c : float
    ptc_canonical_coords : bool
    r0_elec : float
    r0_mag : float
    ref_time_start : float
    slave_status : str
    spin_fringe_on : bool
    static_linear_map : bool
    tilt : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    vkick : float
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Sextupole"]  # = "Sextupole"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    b2_gradient: float = ROField(default=0.0, attr="b2_gradient", tao_name="B2_GRADIENT")
    bl_hkick: float = ROField(default=0.0, attr="bl_hkick", tao_name="BL_HKICK")
    bl_vkick: float = ROField(default=0.0, attr="bl_vkick", tao_name="BL_VKICK")
    csr_ds_step: float = ROField(default=0.0, attr="csr_ds_step", tao_name="CSR_DS_STEP")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    ds_step: float = ROField(default=0.0, attr="ds_step", tao_name="DS_STEP")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    field_master: bool = Field(default=False)
    fringe_at: str = ROField(default="", attr="fringe_at", tao_name="FRINGE_AT")
    fringe_type: str = ROField(default="", attr="fringe_type", tao_name="FRINGE_TYPE")
    hkick: float = ROField(default=0.0, attr="hkick", tao_name="HKICK")
    integrator_order: int = ROField(default=0, attr="integrator_order", tao_name="INTEGRATOR_ORDER")
    k2: float = ROField(default=0.0, attr="k2", tao_name="K2")
    L: float = ROField(default=0.0)
    lord_pad1: float = ROField(default=0.0, attr="lord_pad1", tao_name="LORD_PAD1")
    lord_pad2: float = ROField(default=0.0, attr="lord_pad2", tao_name="LORD_PAD2")
    lord_status: str = ROField(default="")
    num_steps: int = ROField(default=0, attr="num_steps", tao_name="NUM_STEPS")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    ptc_canonical_coords: bool = ROField(
        default=False, attr="ptc_canonical_coords", tao_name="PTC_CANONICAL_COORDS"
    )
    r0_elec: float = ROField(default=0.0, attr="r0_elec", tao_name="R0_ELEC")
    r0_mag: float = ROField(default=0.0, attr="r0_mag", tao_name="R0_MAG")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    slave_status: str = ROField(default="")
    spin_fringe_on: bool = ROField(default=False, attr="spin_fringe_on", tao_name="SPIN_FRINGE_ON")
    static_linear_map: bool = ROField(
        default=False, attr="static_linear_map", tao_name="STATIC_LINEAR_MAP"
    )
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    vkick: float = ROField(default=0.0, attr="vkick", tao_name="VKICK")
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class Sol_QuadAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 45`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    b1_gradient : float
    bl_hkick : float
    bl_vkick : float
    bs_field : float
    csr_ds_step : float
    delta_ref_time : float
    ds_step : float
    e_tot : float
    field_master : bool
    fringe_at : str
    fringe_type : str
    hkick : float
    integrator_order : int
    k1 : float
    ks : float
    L : float
    lord_pad1 : float
    lord_pad2 : float
    lord_status : str
    num_steps : int
    offset_moves_aperture : bool
    p0c : float
    ptc_canonical_coords : bool
    r0_elec : float
    r0_mag : float
    ref_time_start : float
    slave_status : str
    spin_fringe_on : bool
    static_linear_map : bool
    tilt : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    vkick : float
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Sol_Quad"]  # = "Sol_Quad"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    b1_gradient: float = ROField(default=0.0, attr="b1_gradient", tao_name="B1_GRADIENT")
    bl_hkick: float = ROField(default=0.0, attr="bl_hkick", tao_name="BL_HKICK")
    bl_vkick: float = ROField(default=0.0, attr="bl_vkick", tao_name="BL_VKICK")
    bs_field: float = ROField(default=0.0, attr="bs_field", tao_name="BS_FIELD")
    csr_ds_step: float = ROField(default=0.0, attr="csr_ds_step", tao_name="CSR_DS_STEP")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    ds_step: float = ROField(default=0.0, attr="ds_step", tao_name="DS_STEP")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    field_master: bool = Field(default=False)
    fringe_at: str = ROField(default="", attr="fringe_at", tao_name="FRINGE_AT")
    fringe_type: str = ROField(default="", attr="fringe_type", tao_name="FRINGE_TYPE")
    hkick: float = ROField(default=0.0, attr="hkick", tao_name="HKICK")
    integrator_order: int = ROField(default=0, attr="integrator_order", tao_name="INTEGRATOR_ORDER")
    k1: float = ROField(default=0.0, attr="k1", tao_name="K1")
    ks: float = ROField(default=0.0, attr="ks", tao_name="KS")
    L: float = ROField(default=0.0)
    lord_pad1: float = ROField(default=0.0, attr="lord_pad1", tao_name="LORD_PAD1")
    lord_pad2: float = ROField(default=0.0, attr="lord_pad2", tao_name="LORD_PAD2")
    lord_status: str = ROField(default="")
    num_steps: int = ROField(default=0, attr="num_steps", tao_name="NUM_STEPS")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    ptc_canonical_coords: bool = ROField(
        default=False, attr="ptc_canonical_coords", tao_name="PTC_CANONICAL_COORDS"
    )
    r0_elec: float = ROField(default=0.0, attr="r0_elec", tao_name="R0_ELEC")
    r0_mag: float = ROField(default=0.0, attr="r0_mag", tao_name="R0_MAG")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    slave_status: str = ROField(default="")
    spin_fringe_on: bool = ROField(default=False, attr="spin_fringe_on", tao_name="SPIN_FRINGE_ON")
    static_linear_map: bool = ROField(
        default=False, attr="static_linear_map", tao_name="STATIC_LINEAR_MAP"
    )
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    vkick: float = ROField(default=0.0, attr="vkick", tao_name="VKICK")
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class SolenoidAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 43`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    bl_hkick : float
    bl_vkick : float
    bs_field : float
    csr_ds_step : float
    delta_ref_time : float
    ds_step : float
    e_tot : float
    field_master : bool
    fringe_at : str
    fringe_type : str
    hkick : float
    integrator_order : int
    ks : float
    L : float
    l_soft_edge : float
    lord_pad1 : float
    lord_pad2 : float
    lord_status : str
    num_steps : int
    offset_moves_aperture : bool
    p0c : float
    ptc_canonical_coords : bool
    r0_elec : float
    r0_mag : float
    r_solenoid : float
    ref_time_start : float
    slave_status : str
    spin_fringe_on : bool
    static_linear_map : bool
    tilt : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    vkick : float
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Solenoid"]  # = "Solenoid"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    bl_hkick: float = ROField(default=0.0, attr="bl_hkick", tao_name="BL_HKICK")
    bl_vkick: float = ROField(default=0.0, attr="bl_vkick", tao_name="BL_VKICK")
    bs_field: float = ROField(default=0.0, attr="bs_field", tao_name="BS_FIELD")
    csr_ds_step: float = ROField(default=0.0, attr="csr_ds_step", tao_name="CSR_DS_STEP")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    ds_step: float = ROField(default=0.0, attr="ds_step", tao_name="DS_STEP")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    field_master: bool = Field(default=False)
    fringe_at: str = ROField(default="", attr="fringe_at", tao_name="FRINGE_AT")
    fringe_type: str = ROField(default="", attr="fringe_type", tao_name="FRINGE_TYPE")
    hkick: float = ROField(default=0.0, attr="hkick", tao_name="HKICK")
    integrator_order: int = ROField(default=0, attr="integrator_order", tao_name="INTEGRATOR_ORDER")
    ks: float = ROField(default=0.0, attr="ks", tao_name="KS")
    L: float = ROField(default=0.0)
    l_soft_edge: float = ROField(default=0.0, attr="l_soft_edge", tao_name="L_SOFT_EDGE")
    lord_pad1: float = ROField(default=0.0, attr="lord_pad1", tao_name="LORD_PAD1")
    lord_pad2: float = ROField(default=0.0, attr="lord_pad2", tao_name="LORD_PAD2")
    lord_status: str = ROField(default="")
    num_steps: int = ROField(default=0, attr="num_steps", tao_name="NUM_STEPS")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    ptc_canonical_coords: bool = ROField(
        default=False, attr="ptc_canonical_coords", tao_name="PTC_CANONICAL_COORDS"
    )
    r0_elec: float = ROField(default=0.0, attr="r0_elec", tao_name="R0_ELEC")
    r0_mag: float = ROField(default=0.0, attr="r0_mag", tao_name="R0_MAG")
    r_solenoid: float = ROField(default=0.0, attr="r_solenoid", tao_name="R_SOLENOID")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    slave_status: str = ROField(default="")
    spin_fringe_on: bool = ROField(default=False, attr="spin_fringe_on", tao_name="SPIN_FRINGE_ON")
    static_linear_map: bool = ROField(
        default=False, attr="static_linear_map", tao_name="STATIC_LINEAR_MAP"
    )
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    vkick: float = ROField(default=0.0, attr="vkick", tao_name="VKICK")
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class TaylorAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 48`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    delta_e_ref : float
    delta_ref_time : float
    e_tot : float
    L : float
    lord_pad1 : float
    lord_pad2 : float
    lord_status : str
    offset_moves_aperture : bool
    p0c : float
    ref_time_start : float
    slave_status : str
    tilt : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Taylor"]  # = "Taylor"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    delta_e_ref: float = ROField(default=0.0, attr="delta_e_ref", tao_name="DELTA_E_REF")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    L: float = ROField(default=0.0)
    lord_pad1: float = ROField(default=0.0, attr="lord_pad1", tao_name="LORD_PAD1")
    lord_pad2: float = ROField(default=0.0, attr="lord_pad2", tao_name="LORD_PAD2")
    lord_status: str = ROField(default="")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    slave_status: str = ROField(default="")
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class Thick_MultipoleAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 47`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    bl_hkick : float
    bl_vkick : float
    csr_ds_step : float
    delta_ref_time : float
    ds_step : float
    e_tot : float
    field_master : bool
    fringe_at : str
    fringe_type : str
    hkick : float
    integrator_order : int
    L : float
    lord_pad1 : float
    lord_pad2 : float
    lord_status : str
    num_steps : int
    offset_moves_aperture : bool
    p0c : float
    ptc_canonical_coords : bool
    ref_time_start : float
    slave_status : str
    spin_fringe_on : bool
    static_linear_map : bool
    tilt : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    vkick : float
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Thick_Multipole"]  # = "Thick_Multipole"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    bl_hkick: float = ROField(default=0.0, attr="bl_hkick", tao_name="BL_HKICK")
    bl_vkick: float = ROField(default=0.0, attr="bl_vkick", tao_name="BL_VKICK")
    csr_ds_step: float = ROField(default=0.0, attr="csr_ds_step", tao_name="CSR_DS_STEP")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    ds_step: float = ROField(default=0.0, attr="ds_step", tao_name="DS_STEP")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    field_master: bool = Field(default=False)
    fringe_at: str = ROField(default="", attr="fringe_at", tao_name="FRINGE_AT")
    fringe_type: str = ROField(default="", attr="fringe_type", tao_name="FRINGE_TYPE")
    hkick: float = ROField(default=0.0, attr="hkick", tao_name="HKICK")
    integrator_order: int = ROField(default=0, attr="integrator_order", tao_name="INTEGRATOR_ORDER")
    L: float = ROField(default=0.0)
    lord_pad1: float = ROField(default=0.0, attr="lord_pad1", tao_name="LORD_PAD1")
    lord_pad2: float = ROField(default=0.0, attr="lord_pad2", tao_name="LORD_PAD2")
    lord_status: str = ROField(default="")
    num_steps: int = ROField(default=0, attr="num_steps", tao_name="NUM_STEPS")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    ptc_canonical_coords: bool = ROField(
        default=False, attr="ptc_canonical_coords", tao_name="PTC_CANONICAL_COORDS"
    )
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    slave_status: str = ROField(default="")
    spin_fringe_on: bool = ROField(default=False, attr="spin_fringe_on", tao_name="SPIN_FRINGE_ON")
    static_linear_map: bool = ROField(
        default=False, attr="static_linear_map", tao_name="STATIC_LINEAR_MAP"
    )
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    vkick: float = ROField(default=0.0, attr="vkick", tao_name="VKICK")
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class VKickerAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 49`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    bl_kick : float
    csr_ds_step : float
    delta_ref_time : float
    ds_step : float
    e_tot : float
    field_master : bool
    fringe_at : str
    fringe_type : str
    integrator_order : int
    kick : float
    L : float
    lord_pad1 : float
    lord_pad2 : float
    lord_status : str
    num_steps : int
    offset_moves_aperture : bool
    p0c : float
    ptc_canonical_coords : bool
    ref_time_start : float
    slave_status : str
    spin_fringe_on : bool
    static_linear_map : bool
    tilt : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["VKicker"]  # = "VKicker"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    bl_kick: float = ROField(default=0.0, attr="bl_kick", tao_name="BL_KICK")
    csr_ds_step: float = ROField(default=0.0, attr="csr_ds_step", tao_name="CSR_DS_STEP")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    ds_step: float = ROField(default=0.0, attr="ds_step", tao_name="DS_STEP")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    field_master: bool = Field(default=False)
    fringe_at: str = ROField(default="", attr="fringe_at", tao_name="FRINGE_AT")
    fringe_type: str = ROField(default="", attr="fringe_type", tao_name="FRINGE_TYPE")
    integrator_order: int = ROField(default=0, attr="integrator_order", tao_name="INTEGRATOR_ORDER")
    kick: float = ROField(default=0.0, attr="kick", tao_name="KICK")
    L: float = ROField(default=0.0)
    lord_pad1: float = ROField(default=0.0, attr="lord_pad1", tao_name="LORD_PAD1")
    lord_pad2: float = ROField(default=0.0, attr="lord_pad2", tao_name="LORD_PAD2")
    lord_status: str = ROField(default="")
    num_steps: int = ROField(default=0, attr="num_steps", tao_name="NUM_STEPS")
    offset_moves_aperture: bool = Field(default=False)
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    ptc_canonical_coords: bool = ROField(
        default=False, attr="ptc_canonical_coords", tao_name="PTC_CANONICAL_COORDS"
    )
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    slave_status: str = ROField(default="")
    spin_fringe_on: bool = ROField(default=False, attr="spin_fringe_on", tao_name="SPIN_FRINGE_ON")
    static_linear_map: bool = ROField(
        default=False, attr="static_linear_map", tao_name="STATIC_LINEAR_MAP"
    )
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


class WigglerAttributes(TaoAttributesModel):
    """
    Structure which corresponds to Tao `pipe ele:gen_attribs 50`, for example.

    Attributes
    ----------
    aperture_at : str
    aperture_type : str
    b_max : float
    bl_hkick : float
    bl_vkick : float
    csr_ds_step : float
    delta_ref_time : float
    delta_ref_time_user_set : bool
    ds_step : float
    e_tot : float
    field_master : bool
    fringe_at : str
    fringe_type : str
    g_max : float
    hkick : float
    integrator_order : int
    k1x : float
    k1y : float
    kx : float
    L : float
    l_period : float
    lord_pad1 : float
    lord_pad2 : float
    lord_status : str
    n_period : float
    num_steps : int
    offset_moves_aperture : bool
    osc_amplitude : float
    p0c : float
    polarity : float
    ptc_canonical_coords : bool
    r0_elec : float
    r0_mag : float
    ref_time_start : float
    slave_status : str
    spin_fringe_on : bool
    static_linear_map : bool
    tilt : float
    tilt_tot : float
    units : dict[str, str]
        Per-attribute unit information.
    vkick : float
    x1_limit : float
    x2_limit : float
    x_offset : float
    x_offset_tot : float
    x_pitch : float
    x_pitch_tot : float
    y1_limit : float
    y2_limit : float
    y_offset : float
    y_offset_tot : float
    y_pitch : float
    y_pitch_tot : float
    z_offset : float
    z_offset_tot : float
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    key: Literal["Wiggler"]  # = "Wiggler"

    aperture_at: str = Field(default="")
    aperture_type: str = Field(default="")
    b_max: float = ROField(default=0.0, attr="b_max", tao_name="B_MAX")
    bl_hkick: float = ROField(default=0.0, attr="bl_hkick", tao_name="BL_HKICK")
    bl_vkick: float = ROField(default=0.0, attr="bl_vkick", tao_name="BL_VKICK")
    csr_ds_step: float = ROField(default=0.0, attr="csr_ds_step", tao_name="CSR_DS_STEP")
    delta_ref_time: float = ROField(default=0.0, attr="delta_ref_time", tao_name="DELTA_REF_TIME")
    delta_ref_time_user_set: bool = ROField(
        default=False, attr="delta_ref_time_user_set", tao_name="DELTA_REF_TIME_USER_SET"
    )
    ds_step: float = ROField(default=0.0, attr="ds_step", tao_name="DS_STEP")
    e_tot: float = ROField(default=0.0, attr="e_tot", tao_name="E_TOT")
    field_master: bool = Field(default=False)
    fringe_at: str = ROField(default="", attr="fringe_at", tao_name="FRINGE_AT")
    fringe_type: str = ROField(default="", attr="fringe_type", tao_name="FRINGE_TYPE")
    g_max: float = ROField(default=0.0, attr="g_max", tao_name="G_MAX")
    hkick: float = ROField(default=0.0, attr="hkick", tao_name="HKICK")
    integrator_order: int = ROField(default=0, attr="integrator_order", tao_name="INTEGRATOR_ORDER")
    k1x: float = ROField(default=0.0, attr="k1x", tao_name="K1X")
    k1y: float = ROField(default=0.0, attr="k1y", tao_name="K1Y")
    kx: float = ROField(default=0.0, attr="kx", tao_name="KX")
    L: float = ROField(default=0.0)
    l_period: float = ROField(default=0.0, attr="l_period", tao_name="L_PERIOD")
    lord_pad1: float = ROField(default=0.0, attr="lord_pad1", tao_name="LORD_PAD1")
    lord_pad2: float = ROField(default=0.0, attr="lord_pad2", tao_name="LORD_PAD2")
    lord_status: str = ROField(default="")
    n_period: float = ROField(default=0.0, attr="n_period", tao_name="N_PERIOD")
    num_steps: int = ROField(default=0, attr="num_steps", tao_name="NUM_STEPS")
    offset_moves_aperture: bool = Field(default=False)
    osc_amplitude: float = ROField(default=0.0, attr="osc_amplitude", tao_name="OSC_AMPLITUDE")
    p0c: float = ROField(default=0.0, attr="p0c", tao_name="P0C")
    polarity: float = ROField(default=0.0, attr="polarity", tao_name="POLARITY")
    ptc_canonical_coords: bool = ROField(
        default=False, attr="ptc_canonical_coords", tao_name="PTC_CANONICAL_COORDS"
    )
    r0_elec: float = ROField(default=0.0, attr="r0_elec", tao_name="R0_ELEC")
    r0_mag: float = ROField(default=0.0, attr="r0_mag", tao_name="R0_MAG")
    ref_time_start: float = ROField(default=0.0, attr="ref_time_start", tao_name="REF_TIME_START")
    slave_status: str = ROField(default="")
    spin_fringe_on: bool = ROField(default=False, attr="spin_fringe_on", tao_name="SPIN_FRINGE_ON")
    static_linear_map: bool = ROField(
        default=False, attr="static_linear_map", tao_name="STATIC_LINEAR_MAP"
    )
    tilt: float = ROField(default=0.0, attr="tilt", tao_name="TILT")
    tilt_tot: float = ROField(default=0.0, attr="tilt_tot", tao_name="TILT_TOT")
    units: dict[str, str] = ROField(
        default=None, description="Per-attribute unit information.", attr="", tao_name="units"
    )
    vkick: float = ROField(default=0.0, attr="vkick", tao_name="VKICK")
    x1_limit: float = ROField(default=0.0, attr="x1_limit", tao_name="X1_LIMIT")
    x2_limit: float = ROField(default=0.0, attr="x2_limit", tao_name="X2_LIMIT")
    x_offset: float = ROField(default=0.0, attr="x_offset", tao_name="X_OFFSET")
    x_offset_tot: float = ROField(default=0.0, attr="x_offset_tot", tao_name="X_OFFSET_TOT")
    x_pitch: float = ROField(default=0.0, attr="x_pitch", tao_name="X_PITCH")
    x_pitch_tot: float = ROField(default=0.0, attr="x_pitch_tot", tao_name="X_PITCH_TOT")
    y1_limit: float = ROField(default=0.0, attr="y1_limit", tao_name="Y1_LIMIT")
    y2_limit: float = ROField(default=0.0, attr="y2_limit", tao_name="Y2_LIMIT")
    y_offset: float = ROField(default=0.0, attr="y_offset", tao_name="Y_OFFSET")
    y_offset_tot: float = ROField(default=0.0, attr="y_offset_tot", tao_name="Y_OFFSET_TOT")
    y_pitch: float = ROField(default=0.0, attr="y_pitch", tao_name="Y_PITCH")
    y_pitch_tot: float = ROField(default=0.0, attr="y_pitch_tot", tao_name="Y_PITCH_TOT")
    z_offset: float = ROField(default=0.0, attr="z_offset", tao_name="Z_OFFSET")
    z_offset_tot: float = ROField(default=0.0, attr="z_offset_tot", tao_name="Z_OFFSET_TOT")


GeneralAttributes = (
    Beginning_EleAttributes
    | DriftAttributes
    | LcavityAttributes
    | MarkerAttributes
    | ECollimatorAttributes
    | RCollimatorAttributes
    | Crab_CavityAttributes
    | PatchAttributes
    | QuadrupoleAttributes
    | RFCavityAttributes
    | TaylorAttributes
    | SBendAttributes
    | EM_FieldAttributes
    | SolenoidAttributes
    | OctupoleAttributes
    | MatchAttributes
    | SextupoleAttributes
    | MultipoleAttributes
    | ELSeparatorAttributes
    | AB_multipoleAttributes
    | HKickerAttributes
    | InstrumentAttributes
    | KickerAttributes
    | MonitorAttributes
    | Sad_MultAttributes
    | Sol_QuadAttributes
    | VKickerAttributes
    | WigglerAttributes
    | AC_KickerAttributes
    | BeamBeamAttributes
    | FiducialAttributes
    | Floor_ShiftAttributes
    | GKickerAttributes
    | Thick_MultipoleAttributes
    | E_GunAttributes
    | OverlayAttributes
    | GroupAttributes
    | RamperAttributes
    | PipeAttributes
    | GirderAttributes
    | Photon_ForkAttributes
    | CrystalAttributes
    | Photon_InitAttributes
    | ForkAttributes
    | Multilayer_MirrorAttributes
    | DetectorAttributes
    | MaskAttributes
)


key_to_generalattributes: dict[str, type[pydantic.BaseModel]] = {
    "Beginning_Ele": Beginning_EleAttributes,
    "Drift": DriftAttributes,
    "Lcavity": LcavityAttributes,
    "Marker": MarkerAttributes,
    "ECollimator": ECollimatorAttributes,
    "RCollimator": RCollimatorAttributes,
    "Crab_Cavity": Crab_CavityAttributes,
    "Patch": PatchAttributes,
    "Quadrupole": QuadrupoleAttributes,
    "RFCavity": RFCavityAttributes,
    "Taylor": TaylorAttributes,
    "SBend": SBendAttributes,
    "EM_Field": EM_FieldAttributes,
    "Solenoid": SolenoidAttributes,
    "Octupole": OctupoleAttributes,
    "Match": MatchAttributes,
    "Sextupole": SextupoleAttributes,
    "Multipole": MultipoleAttributes,
    "ELSeparator": ELSeparatorAttributes,
    "AB_multipole": AB_multipoleAttributes,
    "HKicker": HKickerAttributes,
    "Instrument": InstrumentAttributes,
    "Kicker": KickerAttributes,
    "Monitor": MonitorAttributes,
    "Sad_Mult": Sad_MultAttributes,
    "Sol_Quad": Sol_QuadAttributes,
    "VKicker": VKickerAttributes,
    "Wiggler": WigglerAttributes,
    "AC_Kicker": AC_KickerAttributes,
    "BeamBeam": BeamBeamAttributes,
    "Fiducial": FiducialAttributes,
    "Floor_Shift": Floor_ShiftAttributes,
    "GKicker": GKickerAttributes,
    "Thick_Multipole": Thick_MultipoleAttributes,
    "E_Gun": E_GunAttributes,
    "Overlay": OverlayAttributes,
    "Group": GroupAttributes,
    "Ramper": RamperAttributes,
    "Pipe": PipeAttributes,
    "Girder": GirderAttributes,
    "Photon_Fork": Photon_ForkAttributes,
    "Crystal": CrystalAttributes,
    "Photon_Init": Photon_InitAttributes,
    "Fork": ForkAttributes,
    "Multilayer_Mirror": Multilayer_MirrorAttributes,
    "Detector": DetectorAttributes,
    "Mask": MaskAttributes,
}

generalattributes_to_key: dict[type[pydantic.BaseModel], str] = {
    Beginning_EleAttributes: "Beginning_Ele",
    DriftAttributes: "Drift",
    LcavityAttributes: "Lcavity",
    MarkerAttributes: "Marker",
    ECollimatorAttributes: "ECollimator",
    RCollimatorAttributes: "RCollimator",
    Crab_CavityAttributes: "Crab_Cavity",
    PatchAttributes: "Patch",
    QuadrupoleAttributes: "Quadrupole",
    RFCavityAttributes: "RFCavity",
    TaylorAttributes: "Taylor",
    SBendAttributes: "SBend",
    EM_FieldAttributes: "EM_Field",
    SolenoidAttributes: "Solenoid",
    OctupoleAttributes: "Octupole",
    MatchAttributes: "Match",
    SextupoleAttributes: "Sextupole",
    MultipoleAttributes: "Multipole",
    ELSeparatorAttributes: "ELSeparator",
    AB_multipoleAttributes: "AB_multipole",
    HKickerAttributes: "HKicker",
    InstrumentAttributes: "Instrument",
    KickerAttributes: "Kicker",
    MonitorAttributes: "Monitor",
    Sad_MultAttributes: "Sad_Mult",
    Sol_QuadAttributes: "Sol_Quad",
    VKickerAttributes: "VKicker",
    WigglerAttributes: "Wiggler",
    AC_KickerAttributes: "AC_Kicker",
    BeamBeamAttributes: "BeamBeam",
    FiducialAttributes: "Fiducial",
    Floor_ShiftAttributes: "Floor_Shift",
    GKickerAttributes: "GKicker",
    Thick_MultipoleAttributes: "Thick_Multipole",
    E_GunAttributes: "E_Gun",
    OverlayAttributes: "Overlay",
    GroupAttributes: "Group",
    RamperAttributes: "Ramper",
    PipeAttributes: "Pipe",
    GirderAttributes: "Girder",
    Photon_ForkAttributes: "Photon_Fork",
    CrystalAttributes: "Crystal",
    Photon_InitAttributes: "Photon_Init",
    ForkAttributes: "Fork",
    Multilayer_MirrorAttributes: "Multilayer_Mirror",
    DetectorAttributes: "Detector",
    MaskAttributes: "Mask",
}
