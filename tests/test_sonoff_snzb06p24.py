"""Tests for the SONOFF SNZB-06P24 presence sensor."""

import asyncio
from unittest import mock

import pytest
from zha.application import Platform
from zha.exceptions import ZHAException
from zha.quirks import DEVICE_REGISTRY
from zigpy.exceptions import DeliveryError
from zigpy.zcl import ClusterType, foundation

from tests.common import ClusterListener
from tests.zha_helpers import join_device_from_diagnostics, zha_gateway
import zhaquirks
from zhaquirks.sonoff.snzb06p24 import (
    SonoffSNZB06P24FC11Cluster,
    SonoffSNZB06P24LocalCluster,
    SonoffZoneSwitch,
    SpatialLearningSubCommand,
    SpatialLearningUiState,
)

zhaquirks.setup()

FC11 = SonoffSNZB06P24FC11Cluster
LOCAL = SonoffSNZB06P24LocalCluster
ZONE_MASK = FC11.AttributeDefs.zone_enable
UI_STATE = LOCAL.AttributeDefs.spatial_learning_ui_state
COUNTDOWN = LOCAL.AttributeDefs.spatial_learning_countdown

DIAGNOSTICS = "sonoff-snzb-06p24-0x00001000.json"
IEEE = "ab:cd:ef:12:8e:52:68:76"
ZONE_SUFFIXES = (
    "zone_0_enable",
    "zone_2_enable",
    "zone_3_enable",
    "zone_4_enable",
    "zone_5_enable",
    "zone_6_enable",
    "zone_7_enable",
)

SEQUENCE = 1_700_000_000_000


@pytest.fixture
def snzb06p24(zigpy_device_from_v2_quirk):
    """SNZB-06P24 device with the quirk applied."""
    return zigpy_device_from_v2_quirk(
        "SONOFF", "SNZB-06P24", cluster_ids={1: {FC11.cluster_id: ClusterType.Server}}
    )


def write_response(status=foundation.Status.SUCCESS):
    """Build a raw write attributes response for the zone mask."""
    if status == foundation.Status.SUCCESS:
        return [[foundation.WriteAttributesStatusRecord(status=status)]]
    return [
        [foundation.WriteAttributesStatusRecord(status=status, attrid=ZONE_MASK.id)]
    ]


async def join_with_zone_mask(gateway, mask):
    """Join the device from its diagnostics, with the given zone mask."""
    device = await join_device_from_diagnostics(gateway, DIAGNOSTICS, DEVICE_REGISTRY)
    device.device.endpoints[1].sonoff_manufacturer.update_attribute(ZONE_MASK.id, mask)
    # Like a restart, after which the mask was read
    await device.recompute_entities()
    return device


def zone_switches(device):
    """Return the zone switches of the device, zone 1 first."""
    return [
        device.get_platform_entity(Platform.SWITCH, unique_id=f"{IEEE}-1-{suffix}")
        for suffix in ZONE_SUFFIXES
    ]


async def test_snzb06p24_zone_switches_need_mask():
    """Test the zone switches are only created once the mask is known."""
    async with zha_gateway() as gateway:
        device = await join_device_from_diagnostics(
            gateway, DIAGNOSTICS, DEVICE_REGISTRY
        )

        assert not any(
            isinstance(entity, SonoffZoneSwitch)
            for entity in device.platform_entities.values()
        )


@pytest.mark.parametrize(
    ("mask", "expected"),
    [
        (0x00FF, (True,) * 7),
        (0x0000, (False,) * 7),
        # Either band of zone 1 is enough for it to be on
        (0x0001, (True,) + (False,) * 6),
        (0x0002, (True,) + (False,) * 6),
        (0x0014, (False, True, False, True, False, False, False)),
    ],
)
async def test_snzb06p24_zone_switches(mask, expected):
    """Test the zone switches follow the zone mask."""
    async with zha_gateway() as gateway:
        device = await join_with_zone_mask(gateway, mask)

        assert tuple(switch.is_on for switch in zone_switches(device)) == expected


@pytest.mark.parametrize(
    ("zone", "state", "mask", "expected_mask"),
    [
        (0, False, 0x01FF, 0x01FC),
        (0, True, 0x01F0, 0x01F3),
        (2, False, 0x01FF, 0x01F7),
        (6, True, 0x0000, 0x0080),
    ],
)
async def test_snzb06p24_zone_write(zone, state, mask, expected_mask):
    """Test a zone switch changes only its bits in the device mask."""
    async with zha_gateway() as gateway:
        device = await join_with_zone_mask(gateway, mask)
        switch = zone_switches(device)[zone]
        fc11 = switch._cluster

        with mock.patch.object(
            fc11, "write_attributes_raw", mock.AsyncMock(return_value=write_response())
        ) as fc11_write:
            await switch.async_turn_on_off(state)

        written = fc11_write.call_args[0][0]
        assert fc11_write.call_args.kwargs["manufacturer_code"] is None
        assert len(written) == 1
        assert written[0].attrid == ZONE_MASK.id
        assert written[0].value.value == expected_mask
        assert fc11.get(ZONE_MASK.id) == expected_mask
        assert switch.is_on is state


async def test_snzb06p24_failed_zone_write():
    """Test a rejected mask write leaves the zone switches untouched."""
    async with zha_gateway() as gateway:
        device = await join_with_zone_mask(gateway, 0x00FF)
        switch = zone_switches(device)[1]
        fc11 = switch._cluster

        with (
            mock.patch.object(
                fc11,
                "write_attributes_raw",
                mock.AsyncMock(return_value=write_response(foundation.Status.FAILURE)),
            ),
            pytest.raises(ZHAException),
        ):
            await switch.async_turn_off()

        assert fc11.get(ZONE_MASK.id) == 0x00FF
        assert switch.is_on is True


async def test_snzb06p24_concurrent_zone_writes():
    """Test concurrent switch changes keep both changes in the device mask."""
    async with zha_gateway() as gateway:
        device = await join_with_zone_mask(gateway, 0x01FF)
        switches = zone_switches(device)
        fc11 = switches[0]._cluster
        entered = asyncio.Event()
        release = asyncio.Event()
        masks = []

        async def delayed_write(attributes, **kwargs):
            masks.append(int(attributes[0].value.value))
            if len(masks) == 1:
                entered.set()
                await release.wait()
            return write_response()

        with mock.patch.object(fc11, "write_attributes_raw", side_effect=delayed_write):
            first = asyncio.create_task(switches[1].async_turn_off())
            await entered.wait()
            second = asyncio.create_task(switches[2].async_turn_off())
            await asyncio.sleep(0)
            release.set()
            await asyncio.gather(first, second)

        assert masks == [0x01FB, 0x01F3]
        assert fc11.get(ZONE_MASK.id) == 0x01F3


@pytest.mark.parametrize("status", [foundation.Status.SUCCESS, None])
async def test_snzb06p24_start_learning(snzb06p24, status):
    """Test the button sends a spatial learning start with the time as sequence."""
    fc11 = snzb06p24.endpoints[1].sonoff_manufacturer
    local = snzb06p24.endpoints[1].sonoff_presence_local
    response = (
        foundation.GENERAL_COMMANDS[foundation.GeneralCommand.Default_Response].schema(
            command_id=FC11.ServerCommandDefs.spatial_learning.id, status=status
        )
        if status is not None
        else None
    )

    with (
        mock.patch.object(
            fc11.endpoint, "request", mock.AsyncMock(return_value=response)
        ) as request,
        mock.patch("time.time", return_value=SEQUENCE / 1000),
    ):
        await fc11.start_learning_now()

    assert local.get(UI_STATE.id) == SpatialLearningUiState.Start
    hdr, data = foundation.ZCLHeader.deserialize(request.call_args.kwargs["data"])
    assert hdr.command_id == FC11.ServerCommandDefs.spatial_learning.id
    assert hdr.frame_control.is_cluster
    assert hdr.manufacturer == snzb06p24.manufacturer_id
    assert data == bytes([SpatialLearningSubCommand.Start]) + SEQUENCE.to_bytes(
        8, "little"
    )

    # The device never answers
    with mock.patch("asyncio.sleep", new_callable=mock.AsyncMock):
        await local._timeout_task

    assert local.get(UI_STATE.id) == SpatialLearningUiState.Idle


# Replies of a SNZB-06P24 to a start request: learning started and takes
# 35 s, then learning succeeded
STARTED_FRAME = bytes.fromhex("117c040152ee4b68a10100000a774c68a1010000")
RESULT_FRAME = bytes.fromhex("11a0040252ee4b68a10100000000")


def send_frame(cluster, frame):
    """Feed a frame from the device to the cluster."""
    cluster.handle_message(*cluster.deserialize(frame))


async def test_snzb06p24_learning_countdown(snzb06p24):
    """Test the countdown follows the duration announced by the device."""
    fc11 = snzb06p24.endpoints[1].sonoff_manufacturer
    local = snzb06p24.endpoints[1].sonoff_presence_local
    listener = ClusterListener(local)

    with mock.patch.object(fc11, "send_default_rsp") as default_rsp:
        send_frame(fc11, STARTED_FRAME)

    assert default_rsp.call_count == 0
    assert local.get(UI_STATE.id) == SpatialLearningUiState.Countdown

    local._timeout_task.cancel()
    with mock.patch("asyncio.sleep", new_callable=mock.AsyncMock):
        await local._countdown_task

    countdown = [
        value for attr, value in listener.attribute_updates if attr == COUNTDOWN.id
    ]
    assert countdown == list(range(35, -1, -1))


async def test_snzb06p24_learning_result(snzb06p24):
    """Test a result from the device ends learning and stops the timers."""
    fc11 = snzb06p24.endpoints[1].sonoff_manufacturer
    local = snzb06p24.endpoints[1].sonoff_presence_local

    with mock.patch.object(fc11, "send_default_rsp"):
        send_frame(fc11, STARTED_FRAME)
        send_frame(fc11, RESULT_FRAME)

    assert local.get(UI_STATE.id) == SpatialLearningUiState.Idle
    assert local.get(COUNTDOWN.id) == 0

    listener = ClusterListener(local)
    with mock.patch("asyncio.sleep", new_callable=mock.AsyncMock):
        await asyncio.gather(
            local._countdown_task, local._timeout_task, return_exceptions=True
        )

    assert local._countdown_task.cancelled()
    assert local._timeout_task.cancelled()
    assert listener.attribute_updates == []


async def test_snzb06p24_learning_timeout(snzb06p24):
    """Test learning gives up when the device never reports a result."""
    fc11 = snzb06p24.endpoints[1].sonoff_manufacturer
    local = snzb06p24.endpoints[1].sonoff_presence_local
    listener = ClusterListener(local)

    with mock.patch.object(fc11, "send_default_rsp"):
        send_frame(fc11, STARTED_FRAME)
    local._countdown_task.cancel()

    with mock.patch("asyncio.sleep", new_callable=mock.AsyncMock):
        await local._timeout_task

    states = [
        value for attr, value in listener.attribute_updates if attr == UI_STATE.id
    ]
    assert states[-2:] == [SpatialLearningUiState.Timeout, SpatialLearningUiState.Idle]
    assert local.get(COUNTDOWN.id) == 0


async def test_snzb06p24_other_commands_untouched(snzb06p24):
    """Test commands other than start_learning_now are sent as is."""
    fc11 = snzb06p24.endpoints[1].sonoff_manufacturer
    local = snzb06p24.endpoints[1].sonoff_presence_local

    with mock.patch.object(fc11.endpoint, "request", mock.AsyncMock()) as request:
        await fc11.command(
            FC11.ServerCommandDefs.spatial_learning.id,
            SpatialLearningSubCommand.Start,
            1,
        )

    _, data = foundation.ZCLHeader.deserialize(request.call_args.kwargs["data"])
    assert data == bytes([SpatialLearningSubCommand.Start]) + (1).to_bytes(8, "little")
    assert local.get(UI_STATE.id) == SpatialLearningUiState.Idle


@pytest.mark.parametrize(
    "frame",
    [
        bytes.fromhex("0101050000"),
        # A spatial learning frame in the other direction
        bytes.fromhex("0901040152ee4b68a10100000a774c68a1010000"),
    ],
    ids=["other_command", "other_direction"],
)
def test_snzb06p24_other_requests_untouched(snzb06p24, frame):
    """Test other cluster requests get the default handling."""
    fc11 = snzb06p24.endpoints[1].sonoff_manufacturer
    local = snzb06p24.endpoints[1].sonoff_presence_local

    with mock.patch.object(fc11, "send_default_rsp") as default_rsp:
        send_frame(fc11, frame)

    assert default_rsp.call_count == 1
    assert local.get(UI_STATE.id) == SpatialLearningUiState.Idle


async def test_snzb06p24_learning_default_response(snzb06p24):
    """Test a default response is sent when the device asks for one."""
    fc11 = snzb06p24.endpoints[1].sonoff_manufacturer
    frame = bytes([STARTED_FRAME[0] & ~0x10]) + STARTED_FRAME[1:]

    with mock.patch.object(fc11, "send_default_rsp") as default_rsp:
        send_frame(fc11, frame)

    assert default_rsp.call_count == 1
    fc11.endpoint.sonoff_presence_local._cancel_tasks()


@pytest.mark.parametrize(
    "error",
    [DeliveryError("no acknowledgement"), TimeoutError(), asyncio.CancelledError()],
)
async def test_snzb06p24_failed_learning_start(snzb06p24, error):
    """A failed or cancelled request restores idle immediately."""
    fc11 = snzb06p24.endpoints[1].sonoff_manufacturer
    local = snzb06p24.endpoints[1].sonoff_presence_local

    with (
        mock.patch.object(fc11.endpoint, "request", mock.AsyncMock(side_effect=error)),
        pytest.raises(type(error)),
    ):
        await fc11.start_learning_now()

    state = local.get(UI_STATE.id)
    cancelling = local._timeout_task.cancelling()
    local._cancel_tasks()
    await asyncio.gather(local._timeout_task, return_exceptions=True)
    assert state == SpatialLearningUiState.Idle
    assert cancelling
    assert local.get(COUNTDOWN.id) == 0


async def test_snzb06p24_rejected_learning_start(snzb06p24):
    """A negative default response ends the pending learning session."""
    fc11 = snzb06p24.endpoints[1].sonoff_manufacturer
    local = snzb06p24.endpoints[1].sonoff_presence_local
    response = foundation.GENERAL_COMMANDS[
        foundation.GeneralCommand.Default_Response
    ].schema(
        command_id=FC11.ServerCommandDefs.spatial_learning.id,
        status=foundation.Status.FAILURE,
    )

    with mock.patch.object(
        fc11.endpoint, "request", mock.AsyncMock(return_value=response)
    ):
        assert await fc11.start_learning_now() == response

    state = local.get(UI_STATE.id)
    cancelling = local._timeout_task.cancelling()
    local._cancel_tasks()
    await asyncio.gather(local._timeout_task, return_exceptions=True)
    assert state == SpatialLearningUiState.Idle
    assert cancelling
