"""Non-finite float results must be rejected, not silently rewritten to null.

JSON cannot represent NaN or Infinity. Serializing them through the default
JSON contract turns them into ``null``, silently corrupting structured content
(#5150). The fix serializes through the real JSON layer and rejects any
non-finite float that survives it, while custom serializers and explicit
per-model policies keep control of their own output.
"""

import math
from typing import Any

import pydantic
import pytest
from pydantic import BaseModel, ConfigDict, Field, field_serializer

from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools.base import ToolResult, _serialize_to_jsonable


class TestNonFiniteStructuredContentRejected:
    async def test_nan_scalar_raises(self):
        mcp = FastMCP()

        @mcp.tool
        def scalar() -> float:
            return float("nan")

        async with Client(mcp) as client:
            with pytest.raises(ToolError, match="non-finite"):
                await client.call_tool("scalar", {})

    async def test_infinity_scalar_raises(self):
        mcp = FastMCP()

        @mcp.tool
        def scalar() -> float:
            return float("inf")

        async with Client(mcp) as client:
            with pytest.raises(ToolError, match="non-finite"):
                await client.call_tool("scalar", {})

    async def test_negative_infinity_scalar_raises(self):
        mcp = FastMCP()

        @mcp.tool
        def scalar() -> float:
            return float("-inf")

        async with Client(mcp) as client:
            with pytest.raises(ToolError, match=r"non-finite float \(-inf\)"):
                await client.call_tool("scalar", {})

    async def test_nan_in_dict_raises_with_value_and_path(self):
        mcp = FastMCP()

        @mcp.tool
        def stats() -> dict[str, float]:
            return {"mean": float("nan"), "count": 2.0}

        async with Client(mcp) as client:
            with pytest.raises(ToolError, match=r"nan.*\$\.mean"):
                await client.call_tool("stats", {})

    async def test_nan_in_list_raises(self):
        mcp = FastMCP()

        @mcp.tool
        def values() -> list[float]:
            return [1.0, float("nan")]

        async with Client(mcp) as client:
            with pytest.raises(ToolError, match="non-finite"):
                await client.call_tool("values", {})

    async def test_nan_in_nested_containers_raises(self):
        mcp = FastMCP()

        @mcp.tool
        def nested() -> dict[str, list[dict[str, float]]]:
            return {"rows": [{"v": 1.0}, {"v": float("inf")}]}

        async with Client(mcp) as client:
            with pytest.raises(ToolError, match=r"non-finite.*\$\.rows\[1\]\.v"):
                await client.call_tool("nested", {})

    async def test_nan_in_plain_base_model_raises(self):
        class Stats(BaseModel):
            mean: float
            count: int = 1

        mcp = FastMCP()

        @mcp.tool
        def stats() -> Stats:
            return Stats(mean=float("nan"))

        async with Client(mcp) as client:
            with pytest.raises(ToolError, match=r"nan.*\$\.mean"):
                await client.call_tool("stats", {})

    @pytest.mark.parametrize("bad", [float("inf"), float("-inf")])
    async def test_inf_in_plain_base_model_raises(self, bad: float):
        class Stats(BaseModel):
            mean: float

        mcp = FastMCP()

        @mcp.tool
        def stats() -> Stats:
            return Stats(mean=bad)

        async with Client(mcp) as client:
            with pytest.raises(ToolError, match=r"non-finite.*\$\.mean"):
                await client.call_tool("stats", {})

    async def test_nan_in_untyped_base_model_return_raises(self):
        class Stats(BaseModel):
            mean: float

        mcp = FastMCP()

        @mcp.tool
        def stats():
            return Stats(mean=float("nan"))

        async with Client(mcp) as client:
            with pytest.raises(ToolError, match=r"nan.*\$\.mean"):
                await client.call_tool("stats", {})

    async def test_nan_in_nested_base_model_list_raises_with_path(self):
        class Stats(BaseModel):
            mean: float

        class Outer(BaseModel):
            rows: list[Stats]

        mcp = FastMCP()

        @mcp.tool
        def report() -> Outer:
            return Outer(rows=[Stats(mean=1.0), Stats(mean=float("inf"))])

        async with Client(mcp) as client:
            with pytest.raises(ToolError, match=r"non-finite.*\$\.rows\[1\]\.mean"):
                await client.call_tool("report", {})

    async def test_nan_in_pydantic_dataclass_raises(self):
        @pydantic.dataclasses.dataclass
        class StatsDC:
            mean: float

        mcp = FastMCP()

        @mcp.tool
        def stats() -> StatsDC:
            return StatsDC(mean=float("nan"))

        async with Client(mcp) as client:
            with pytest.raises(ToolError, match=r"nan.*\$\.mean"):
                await client.call_tool("stats", {})


class TestFiniteValuesUnaffected:
    async def test_finite_structured_content_round_trips(self):
        mcp = FastMCP()

        @mcp.tool
        def stats() -> dict[str, float]:
            return {"mean": 2.5, "count": 2.0}

        async with Client(mcp) as client:
            result = await client.call_tool("stats", {})

        assert result.structured_content == {"mean": 2.5, "count": 2.0}

    async def test_nan_converted_by_custom_field_serializer_is_preserved(self):
        class Report(BaseModel):
            mean: float

            @field_serializer("mean")
            def _mean(self, value: float) -> str:
                return f"{value:.4f}"

        mcp = FastMCP()

        @mcp.tool
        def report() -> Report:
            return Report(mean=float("nan"))

        async with Client(mcp) as client:
            result = await client.call_tool("report", {})

        # The custom serializer deliberately converted the value to valid
        # JSON, so it must pass through untouched.
        assert result.structured_content == {"mean": "nan"}


class TestSerializeToJsonableUnit:
    def test_any_fallback_rejects_non_finite(self):
        with pytest.raises(ToolError, match="non-finite"):
            _serialize_to_jsonable({"v": float("nan")})

    def test_any_fallback_rejects_non_finite_nested(self):
        with pytest.raises(ToolError, match=r"non-finite.*\$\.a\[1\]"):
            _serialize_to_jsonable({"a": [1.0, math.inf]})

    def test_typed_annotation_rejects_non_finite(self):
        with pytest.raises(ToolError, match="non-finite"):
            _serialize_to_jsonable({"v": float("nan")}, dict[str, float])

    def test_finite_values_pass_through(self):
        value = {"a": [1.0, 2.5], "b": None, "c": "x"}
        assert _serialize_to_jsonable(value, dict[str, Any]) == value

    def test_string_mentioning_nan_is_not_rejected(self):
        value = {"note": "the value was NaN at index 0", "data": [1.0]}
        assert _serialize_to_jsonable(value) == value

    def test_explicit_model_null_policy_is_preserved(self):
        class Lenient(BaseModel):
            model_config = ConfigDict(ser_json_inf_nan="null")
            mean: float

        # An explicit per-model policy deliberately emits null, which is valid
        # JSON — the boundary must not second-guess it.
        assert _serialize_to_jsonable(Lenient(mean=float("nan")), Lenient) == {
            "mean": None
        }

    def test_explicit_model_strings_policy_is_preserved(self):
        class StringPol(BaseModel):
            model_config = ConfigDict(ser_json_inf_nan="strings")
            mean: float

        assert _serialize_to_jsonable(StringPol(mean=float("nan")), StringPol) == {
            "mean": "NaN"
        }

    def test_nested_explicit_null_policy_inside_plain_outer_model(self):
        class Lenient(BaseModel):
            model_config = ConfigDict(ser_json_inf_nan="null")
            mean: float

        class Outer(BaseModel):
            ok: float
            lenient: Lenient

        assert _serialize_to_jsonable(
            Outer(ok=1.0, lenient=Lenient(mean=float("nan"))), Outer
        ) == {"ok": 1.0, "lenient": {"mean": None}}

        with pytest.raises(ToolError, match=r"nan.*\$\.ok"):
            _serialize_to_jsonable(
                Outer(ok=float("nan"), lenient=Lenient(mean=float("nan"))), Outer
            )

    def test_excluded_nonfinite_field_on_plain_base_model_is_ignored(self):
        class SecretNan(BaseModel):
            ok: float
            hidden: float = Field(default=float("nan"), exclude=True)

        assert _serialize_to_jsonable(SecretNan(ok=1.5), SecretNan) == {"ok": 1.5}
        assert _serialize_to_jsonable(SecretNan(ok=1.5)) == {"ok": 1.5}

    def test_any_container_with_plain_and_recursive_models_rejects_nonfinite(self):
        class Node(BaseModel):
            val: float
            children: list["Node"] = []

        Node.model_rebuild()

        with pytest.raises(ToolError, match=r"non-finite.*\$\.children\[0\]\.val"):
            _serialize_to_jsonable(Node(val=1.0, children=[Node(val=math.inf)]), Node)

        with pytest.raises(ToolError, match=r"nan.*\$\.node\.val"):
            _serialize_to_jsonable({"node": Node(val=float("nan"))}, dict[str, Any])

    def test_tool_result_structured_content_rejects_non_finite(self):
        with pytest.raises(ToolError, match="non-finite"):
            ToolResult(structured_content={"v": float("nan")})


class TestDocumentedResidualBehavior:
    """Documented trade-offs that remain by design (see follow-up issues).

    These tests pin the *current* behavior so any future policy change is a
    conscious, reviewed decision rather than a silent drift.
    """

    def test_default_serializer_rewrites_non_finite_to_null(self):
        """`default_serializer` (the text-content path) still nulls non-finites.

        It goes through pydantic's default JSON policy, so it can disagree with
        the structured path this PR fixed. Tracked as a follow-up.
        """
        from fastmcp.tools.base import default_serializer

        assert default_serializer({"mean": float("nan")}) == '{"mean":null}'

    async def test_unserializable_result_falls_back_to_text_content_softly(self):
        """A value the structured serializer cannot handle becomes content-only.

        `convert_result` catches the serialization error and returns text
        content with `is_error` false — a soft success. This is the leftover
        text-path risk: a non-finite value that fails structured serialization
        surfaces as text, never as a tool error (follow-up: fail the text path
        too, e.g. an `inf_nan_mode="error"`-style check).
        """

        class Unserializable:
            def __str__(self):
                return "nan"

        mcp = FastMCP()

        @mcp.tool
        def weird():
            return Unserializable()

        async with Client(mcp) as client:
            result = await client.call_tool("weird", {})

        assert result.is_error is False
        assert result.structured_content is None
        # The text content went through the default JSON serializer, which
        # stringifies the value ("nan" as a JSON string), with no error raised.
        assert result.content[0].text == '"nan"'
