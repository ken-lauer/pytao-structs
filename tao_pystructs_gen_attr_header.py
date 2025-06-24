#!/usr/bin/env python
# vi: syntax=python sw=4 ts=4 sts=4
"""
This file is auto-generated; do not hand-edit it.
"""

from __future__ import annotations

import contextlib
import functools  # noqa: F401
import logging
import textwrap
from typing import (
    cast,
    Annotated,  # noqa: F401
    Any,
    ClassVar,
    Literal,  # noqa: F401
)

import numpy as np
import pydantic  # noqa: F401
from pytao import Tao
from rich.pretty import pretty_repr
from typing_extensions import Self


# TODO: this code is partly duplicated with TaoSettableModel; refactor
logger = logging.getLogger(__name__)


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


class TaoAttributesModel(
    pydantic.BaseModel,
    validate_assignment=True,
    extra="allow",  # NOTE: differs from the rest of pystructs
):
    """
    A helper base class which allows for creating/updating an instance with Tao objects.
    """

    # The `Tao.cmd_attr` command to query this information.
    _tao_command_attr_: ClassVar[str]
    # Default arguments to pass to `Tao.cmd_attr(**default_args)`
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

        key = generalattributes_to_key[cls]  # noqa: F821
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
