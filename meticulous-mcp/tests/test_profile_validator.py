"""Tests for profile validator."""

import json
import tempfile
from pathlib import Path

import pytest

from meticulous_mcp.profile_validator import ProfileValidationError, ProfileValidator, ValidationLevel


@pytest.fixture
def sample_schema():
    """Create a sample schema file."""
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "id": {"type": "string"},
            "temperature": {"type": "number", "minimum": 0, "maximum": 100},
        },
        "required": ["name", "id", "temperature"],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(schema, f)
        schema_path = f.name
    
    # File is closed but not deleted, yield the path
    yield schema_path
    # Clean up after test
    if Path(schema_path).exists():
        Path(schema_path).unlink()


@pytest.fixture
def validator(sample_schema):
    """Create a validator instance."""
    return ProfileValidator(schema_path=sample_schema)


def test_valid_profile(validator):
    """Test validation of a valid profile."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
    }
    is_valid, errors = validator.validate(profile)
    assert is_valid
    assert len(errors) == 0


def test_invalid_profile_missing_field(validator):
    """Test validation of profile with missing required field."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        # Missing temperature
    }
    is_valid, errors = validator.validate(profile)
    assert not is_valid
    assert len(errors) > 0


def test_invalid_profile_wrong_type(validator):
    """Test validation of profile with wrong type."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": "not-a-number",  # Should be number
    }
    is_valid, errors = validator.validate(profile)
    assert not is_valid
    assert len(errors) > 0


def test_validate_and_raise_valid(validator):
    """Test validate_and_raise with valid profile."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
    }
    # Should not raise
    validator.validate_and_raise(profile)


def test_validate_and_raise_invalid(validator):
    """Test validate_and_raise with invalid profile."""
    profile = {
        "name": "Test Profile",
        # Missing required fields
    }
    with pytest.raises(ProfileValidationError) as exc_info:
        validator.validate_and_raise(profile)
    assert len(exc_info.value.errors) > 0


def test_lint_valid_profile(validator):
    """Test linting of a valid profile."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [{"name": "Stage 1", "exit_triggers": [{"type": "time", "value": 30}]}],
    }
    warnings = validator.lint(profile)
    # Should have no warnings for a well-formed profile
    assert isinstance(warnings, list)


def test_lint_profile_no_stages(validator):
    """Test linting of profile without stages."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [],
    }
    warnings = validator.lint(profile)
    assert any("no stages" in w.lower() for w in warnings)


def test_lint_profile_unusual_temperature(validator):
    """Test linting of profile with unusual temperature."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 150.0,  # Too high
    }
    warnings = validator.lint(profile)
    assert any("temperature" in w.lower() for w in warnings)


def test_validator_init_with_path(sample_schema):
    """Test validator initialization with explicit path."""
    validator = ProfileValidator(schema_path=sample_schema)
    assert validator is not None


def test_validator_init_no_schema():
    """Test validator initialization without schema file."""
    with pytest.raises(FileNotFoundError):
        ProfileValidator(schema_path="/nonexistent/path/schema.json")


def test_validation_error_message_formatting(validator):
    """Test that validation errors are properly formatted."""
    profile = {
        "name": "Test Profile",
        # Missing id and temperature
    }
    is_valid, errors = validator.validate(profile)
    assert not is_valid
    assert len(errors) > 0
    # Check that errors are formatted strings
    for error in errors:
        assert isinstance(error, str)
        assert len(error) > 0


def test_validate_multiple_errors(validator):
    """Test that multiple validation errors are collected."""
    profile = {
        "name": "Test Profile",
        # Missing id and temperature
        "temperature": "not-a-number",  # Wrong type
    }
    is_valid, errors = validator.validate(profile)
    assert not is_valid
    assert len(errors) >= 2  # At least 2 errors


def test_validate_and_raise_error_message(validator):
    """Test that validate_and_raise includes all errors in message."""
    profile = {
        "name": "Test Profile",
        # Missing id and temperature
    }
    with pytest.raises(ProfileValidationError) as exc_info:
        validator.validate_and_raise(profile)
    
    error = exc_info.value
    assert len(error.errors) > 0
    # Check that the exception message includes all errors
    assert str(error)
    assert len(str(error)) > len(error.message)  # Should include formatted errors


def test_lint_stage_no_exit_triggers(validator):
    """Test linting of stage without exit triggers."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {"points": [[0, 4]], "over": "time"},
                "exit_triggers": [],  # No exit triggers
            }
        ],
    }
    warnings = validator.lint(profile)
    assert any("no exit triggers" in w.lower() for w in warnings)


def test_lint_stage_no_weight_or_time_trigger(validator):
    """Test linting of stage without weight or time triggers."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {"points": [[0, 4]], "over": "time"},
                "exit_triggers": [{"type": "pressure", "value": 9.0}],  # No weight/time
            }
        ],
    }
    warnings = validator.lint(profile)
    assert any("none are weight or time-based" in w.lower() for w in warnings)


def test_lint_stage_empty_dynamics_points(validator):
    """Test linting of stage with empty dynamics points."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {"points": [], "over": "time"},
                "exit_triggers": [{"type": "time", "value": 30}],
            }
        ],
    }
    warnings = validator.lint(profile)
    assert any("empty dynamics points" in w.lower() for w in warnings)


def test_lint_stage_single_dynamics_point(validator):
    """Test linting of stage with only one dynamics point."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {"points": [[0, 4]], "over": "time"},
                "exit_triggers": [{"type": "time", "value": 30}],
            }
        ],
    }
    warnings = validator.lint(profile)
    assert any("only one dynamics point" in w.lower() for w in warnings)


def test_lint_stage_invalid_dynamics_over(validator):
    """Test linting of stage with invalid dynamics.over value."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {"points": [[0, 4]], "over": "invalid"},
                "exit_triggers": [{"type": "time", "value": 30}],
            }
        ],
    }
    warnings = validator.lint(profile)
    assert any("invalid dynamics.over" in w.lower() for w in warnings)


def test_lint_stage_invalid_type(validator):
    """Test linting of stage with invalid type."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "invalid_type",
                "dynamics": {"points": [[0, 4]], "over": "time"},
                "exit_triggers": [{"type": "time", "value": 30}],
            }
        ],
    }
    warnings = validator.lint(profile)
    assert any("invalid type" in w.lower() for w in warnings)


def test_lint_duplicate_stage_keys(validator):
    """Test linting of profile with duplicate stage keys."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {"points": [[0, 4]], "over": "time"},
                "exit_triggers": [{"type": "time", "value": 30}],
            },
            {
                "name": "Stage 2",
                "key": "stage_1",  # Duplicate key
                "type": "flow",
                "dynamics": {"points": [[0, 4]], "over": "time"},
                "exit_triggers": [{"type": "time", "value": 30}],
            },
        ],
    }
    warnings = validator.lint(profile)
    assert any("duplicate key" in w.lower() for w in warnings)


def test_lint_temperature_low(validator):
    """Test linting of profile with low temperature."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 82.0,  # Low
    }
    warnings = validator.lint(profile)
    assert any("lower end" in w.lower() for w in warnings)


def test_lint_temperature_high(validator):
    """Test linting of profile with high temperature."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 97.0,  # High
    }
    warnings = validator.lint(profile)
    assert any("higher end" in w.lower() for w in warnings)


def test_lint_final_weight_low(validator):
    """Test linting of profile with low final weight."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "final_weight": 15.0,  # Low
    }
    warnings = validator.lint(profile)
    assert any("quite low" in w.lower() for w in warnings)


def test_lint_final_weight_high(validator):
    """Test linting of profile with high final weight."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "final_weight": 75.0,  # High
    }
    warnings = validator.lint(profile)
    assert any("quite high" in w.lower() for w in warnings)


def test_lint_final_weight_out_of_range(validator):
    """Test linting of profile with final weight out of typical range."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "final_weight": 150.0,  # Way too high
    }
    warnings = validator.lint(profile)
    assert any("outside typical range" in w.lower() for w in warnings)


def test_lint_undefined_variable_reference(validator):
    """Test linting of profile with undefined variable reference."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {
                    "points": [[0, "$undefined_var"]],  # Undefined variable
                    "over": "time",
                },
                "exit_triggers": [{"type": "time", "value": 30}],
            }
        ],
        "variables": [{"name": "Other", "key": "other_var", "type": "pressure", "value": 8.0}],  # Different variable defined
    }
    warnings = validator.lint(profile)
    # Check that undefined variable reference is detected
    assert any("references variable" in w.lower() and "undefined_var" in w.lower() for w in warnings)


def test_lint_stages_not_list(validator):
    """Test linting when stages is not a list."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": "not-a-list",  # Wrong type
    }
    warnings = validator.lint(profile)
    assert any("should be a list" in w.lower() for w in warnings)


def test_lint_variables_not_list(validator):
    """Test linting when variables is not a list (should handle gracefully)."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "variables": "not-a-list",
        "stages": [],
    }
    # Should not crash, warnings may or may not be generated
    warnings = validator.lint(profile)
    assert isinstance(warnings, list)


def test_lint_stage_not_dict(validator):
    """Test linting when stage is not a dict (should handle gracefully)."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": ["not-a-dict", {"name": "Valid Stage", "exit_triggers": []}],
    }
    warnings = validator.lint(profile)
    assert isinstance(warnings, list)


def test_validation_error_exception_properties(validator):
    """Test ProfileValidationError exception properties."""
    profile = {
        "name": "Test Profile",
        # Missing id and temperature
    }
    with pytest.raises(ProfileValidationError) as exc_info:
        validator.validate_and_raise(profile)
    
    error = exc_info.value
    assert hasattr(error, 'message')
    assert hasattr(error, 'errors')
    assert isinstance(error.errors, list)
    assert len(error.errors) > 0


def test_format_error_with_path(validator):
    """Test error formatting includes path information."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": "not-a-number",  # Wrong type
    }
    is_valid, errors = validator.validate(profile)
    assert not is_valid
    # Check that errors include field path information
    assert any("temperature" in error.lower() for error in errors)


def test_lint_empty_profile(validator):
    """Test linting of empty profile."""
    profile = {}
    warnings = validator.lint(profile)
    assert isinstance(warnings, list)


def test_validate_empty_profile(validator):
    """Test validation of empty profile."""
    profile = {}
    is_valid, errors = validator.validate(profile)
    assert not is_valid
    assert len(errors) > 0


def test_lint_missing_limits_field(validator):
    """Test linting detects missing limits field."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {"points": [[0, 4]], "over": "time"},
                "exit_triggers": [{"type": "time", "value": 30}],
                # Missing limits field
            }
        ],
    }
    warnings = validator.lint(profile)
    assert any("missing 'limits' field" in w.lower() for w in warnings)
    assert any("limits" in w.lower() for w in warnings)


def test_lint_null_limits_field(validator):
    """Test linting detects null limits field."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {"points": [[0, 4]], "over": "time"},
                "exit_triggers": [{"type": "time", "value": 30}],
                "limits": None,  # null limits
            }
        ],
    }
    warnings = validator.lint(profile)
    assert any("limits" in w.lower() and "null" in w.lower() for w in warnings)


def test_lint_missing_relative_field(validator):
    """Test linting detects missing relative field in exit triggers."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {"points": [[0, 4]], "over": "time"},
                "exit_triggers": [
                    {"type": "time", "value": 30},  # Missing relative field
                    {"type": "weight", "value": 40, "relative": False},  # Has relative
                ],
            }
        ],
    }
    warnings = validator.lint(profile)
    assert any("missing 'relative' field" in w.lower() for w in warnings)
    assert any("relative" in w.lower() for w in warnings)


def test_lint_null_relative_field(validator):
    """Test linting detects null relative field in exit triggers."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {"points": [[0, 4]], "over": "time"},
                "exit_triggers": [
                    {"type": "time", "value": 30, "relative": None},  # null relative
                ],
            }
        ],
    }
    warnings = validator.lint(profile)
    assert any("missing 'relative' field" in w.lower() or ("relative" in w.lower() and "null" in w.lower()) for w in warnings)


def test_lint_multiple_normalization_issues(validator):
    """Test linting detects multiple normalization issues."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {"points": [[0, 4]], "over": "time"},
                "exit_triggers": [
                    {"type": "time", "value": 30},  # Missing relative
                ],
                "limits": None,  # null limits
            },
            {
                "name": "Stage 2",
                "key": "stage_2",
                "type": "pressure",
                "dynamics": {"points": [[0, 8]], "over": "time"},
                "exit_triggers": [
                    {"type": "weight", "value": 40},  # Missing relative
                ],
                # Missing limits field entirely
            },
        ],
    }
    warnings = validator.lint(profile)
    # Should have warnings for both stages
    limits_warnings = [w for w in warnings if "limits" in w.lower()]
    relative_warnings = [w for w in warnings if "relative" in w.lower()]
    assert len(limits_warnings) >= 2  # At least 2 limits warnings
    assert len(relative_warnings) >= 2  # At least 2 relative warnings


def test_lint_no_warnings_when_fields_present(validator):
    """Test linting doesn't warn when fields are present with valid values."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {"points": [[0, 4]], "over": "time"},
                "exit_triggers": [
                    {"type": "time", "value": 30, "relative": False},  # Has relative
                ],
                "limits": [],  # Has limits (empty array is valid)
            }
        ],
    }
    warnings = validator.lint(profile)
    # Should not have warnings about missing limits or relative
    limits_warnings = [w for w in warnings if "missing 'limits' field" in w.lower() or ("limits" in w.lower() and "null" in w.lower())]
    relative_warnings = [w for w in warnings if "missing 'relative' field" in w.lower()]
    assert len(limits_warnings) == 0
    assert len(relative_warnings) == 0


def test_validate_pressure_exceeds_15_bar_in_dynamics(validator):
    """Test validation fails when pressure exceeds 15 bar in dynamics points."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "High Pressure Stage",
                "key": "stage_1",
                "type": "pressure",  # Pressure-type stage
                "dynamics": {
                    "points": [[0, 9], [30, 20]],  # 20 bar exceeds limit
                    "over": "time",
                },
                "exit_triggers": [{"type": "time", "value": 30, "relative": False}],
                "limits": [],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    assert not is_valid
    assert any("pressure" in e.lower() and "20" in e and "15 bar limit" in e.lower() for e in errors)


def test_validate_negative_pressure_in_dynamics(validator):
    """Test validation fails when pressure is negative in dynamics points."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Negative Pressure Stage",
                "key": "stage_1",
                "type": "pressure",
                "dynamics": {
                    "points": [[0, 9], [30, -5]],  # Negative pressure
                    "over": "time",
                },
                "exit_triggers": [{"type": "time", "value": 30, "relative": False}],
                "limits": [],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    assert not is_valid
    assert any("negative pressure" in e.lower() and "-5" in e for e in errors)


def test_validate_pressure_exceeds_15_bar_in_exit_trigger(validator):
    """Test validation fails when pressure exceeds 15 bar in exit triggers."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {"points": [[0, 4], [30, 4]], "over": "time"},
                "exit_triggers": [
                    {"type": "pressure", "value": 18, "relative": False, "comparison": ">="}  # 18 bar exceeds limit
                ],
                "limits": [],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    assert not is_valid
    assert any("exit trigger" in e.lower() and "pressure" in e.lower() and "18" in e and "15 bar limit" in e.lower() for e in errors)


def test_validate_negative_pressure_in_exit_trigger(validator):
    """Test validation fails when pressure is negative in exit triggers."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {"points": [[0, 4], [30, 4]], "over": "time"},
                "exit_triggers": [
                    {"type": "pressure", "value": -2, "relative": False, "comparison": ">="}  # Negative pressure
                ],
                "limits": [],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    assert not is_valid
    assert any("exit trigger" in e.lower() and "negative pressure" in e.lower() and "-2" in e for e in errors)


def test_validate_pressure_validation_only_for_pressure_stages(validator):
    """Test that pressure validation in dynamics only applies to pressure-type stages."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Flow Stage",
                "key": "stage_1",
                "type": "flow",  # Flow-type stage, not pressure
                "dynamics": {
                    "points": [[0, 20], [30, 25]],  # These are flow values, not pressure
                    "over": "time",
                },
                "exit_triggers": [{"type": "time", "value": 30, "relative": False}],
                "limits": [],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    # Should be valid - no pressure errors since this is a flow stage
    pressure_errors = [e for e in errors if "15 bar limit" in e.lower()]
    assert len(pressure_errors) == 0


def test_validate_pressure_limits_not_validated(validator):
    """Test that pressure in limits is NOT validated (can be infinity)."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {"points": [[0, 4], [30, 4]], "over": "time"},
                "exit_triggers": [{"type": "time", "value": 30, "relative": False}],
                "limits": [
                    {"type": "pressure", "value": 100}  # Very high pressure limit - this is OK
                ],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    # Should NOT have errors about pressure in limits
    limits_pressure_errors = [e for e in errors if "limit" in e.lower() and "pressure" in e.lower() and "15 bar" in e.lower()]
    assert len(limits_pressure_errors) == 0


def test_validate_valid_pressure_values(validator):
    """Test that valid pressure values (within 0-15 bar) pass validation."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Valid Pressure Stage",
                "key": "stage_1",
                "type": "pressure",
                "dynamics": {
                    "points": [[0, 2], [10, 9], [20, 6]],  # All within 0-15 bar
                    "over": "time",
                },
                "exit_triggers": [
                    {"type": "pressure", "value": 8, "relative": False, "comparison": ">="},  # Within limit
                    {"type": "time", "value": 30, "relative": False},
                ],
                "limits": [],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    # Should NOT have any pressure limit errors
    pressure_errors = [e for e in errors if "15 bar limit" in e.lower() or "negative pressure" in e.lower()]
    assert len(pressure_errors) == 0


def test_validate_multiple_pressure_violations(validator):
    """Test validation detects multiple pressure violations in different locations."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "pressure",
                "dynamics": {
                    "points": [[0, 20], [30, -3]],  # Both exceed limit and negative
                    "over": "time",
                },
                "exit_triggers": [
                    {"type": "pressure", "value": 18, "relative": False, "comparison": ">="}  # Also exceeds
                ],
                "limits": [],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    assert not is_valid
    # Should have at least 3 errors (2 from dynamics, 1 from exit trigger)
    pressure_errors = [e for e in errors if ("15 bar limit" in e.lower() or "negative pressure" in e.lower())]
    assert len(pressure_errors) >= 3


def test_validate_interpolation_none_fails(validator):
    """Test validation fails when interpolation is 'none' (not supported by machine)."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Stage with none interpolation",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {
                    "points": [[0, 4]],
                    "over": "time",
                    "interpolation": "none",  # This should fail
                },
                "exit_triggers": [{"type": "time", "value": 30, "relative": True}],
                "limits": [],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    assert not is_valid
    assert any("interpolation" in e.lower() and "none" in e.lower() and "not supported" in e.lower() for e in errors)


def test_validate_interpolation_linear_passes(validator):
    """Test validation passes when interpolation is 'linear'."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Stage with linear interpolation",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {
                    "points": [[0, 4], [10, 6]],
                    "over": "time",
                    "interpolation": "linear",
                },
                "exit_triggers": [{"type": "time", "value": 30, "relative": True}],
                "limits": [{"type": "pressure", "value": 10}],
            }
        ],
    }
    is_valid, errors = validator.validate(profile, level=ValidationLevel.MACHINE)
    assert is_valid
    interpolation_errors = [e for e in errors if "interpolation" in e.lower()]
    assert len(interpolation_errors) == 0


def test_validate_interpolation_curve_passes(validator):
    """Test validation passes when interpolation is 'curve'."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Stage with curve interpolation",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {
                    "points": [[0, 4], [10, 6]],
                    "over": "time",
                    "interpolation": "curve",
                },
                "exit_triggers": [{"type": "time", "value": 30, "relative": True}],
                "limits": [{"type": "pressure", "value": 10}],
            }
        ],
    }
    is_valid, errors = validator.validate(profile, level=ValidationLevel.MACHINE)
    assert is_valid
    interpolation_errors = [e for e in errors if "interpolation" in e.lower()]
    assert len(interpolation_errors) == 0


def test_validate_interpolation_invalid_value_fails(validator):
    """Test validation fails for any invalid interpolation value."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Stage with invalid interpolation",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {
                    "points": [[0, 4]],
                    "over": "time",
                    "interpolation": "invalid_value",  # This should fail
                },
                "exit_triggers": [{"type": "time", "value": 30, "relative": True}],
                "limits": [],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    assert not is_valid
    assert any("interpolation" in e.lower() for e in errors)


def test_lint_interpolation_none_warns(validator):
    """Test linting warns when interpolation is 'none'."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "dynamics": {
                    "points": [[0, 4]],
                    "over": "time",
                    "interpolation": "none",
                },
                "exit_triggers": [{"type": "time", "value": 30}],
            }
        ],
    }
    warnings = validator.lint(profile)
    assert any("interpolation" in w.lower() and "none" in w.lower() and "not supported" in w.lower() for w in warnings)


# ==================== Additional Validation Tests ====================

def test_validate_curve_interpolation_requires_two_points(validator):
    """Test validation fails when curve interpolation has only 1 point."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Single Point Curve",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {
                    "points": [[0, 4]],  # Only 1 point with curve
                    "over": "time",
                    "interpolation": "curve",
                },
                "exit_triggers": [{"type": "time", "value": 30, "relative": True}],
                "limits": [],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    assert not is_valid
    assert any("curve" in e.lower() and "2 points" in e.lower() for e in errors)


def test_validate_curve_interpolation_with_two_points_passes(validator):
    """Test validation passes when curve interpolation has 2+ points."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Multi Point Curve",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {
                    "points": [[0, 4], [10, 6]],  # 2 points - OK for curve
                    "over": "time",
                    "interpolation": "curve",
                },
                "exit_triggers": [{"type": "time", "value": 30, "relative": True}],
                "limits": [],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    curve_errors = [e for e in errors if "curve" in e.lower() and "2 points" in e.lower()]
    assert len(curve_errors) == 0


def test_validate_invalid_dynamics_over_fails(validator):
    """Test validation fails for invalid dynamics.over values."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Invalid Over",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {
                    "points": [[0, 4]],
                    "over": "invalid_value",  # Invalid
                    "interpolation": "linear",
                },
                "exit_triggers": [{"type": "time", "value": 30, "relative": True}],
                "limits": [],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    assert not is_valid
    assert any("dynamics.over" in e.lower() and "invalid_value" in e for e in errors)


def test_validate_valid_dynamics_over_passes(validator):
    """Test validation passes for valid dynamics.over values."""
    for over_value in ["time", "weight", "piston_position"]:
        profile = {
            "name": "Test Profile",
            "id": "test-id",
            "temperature": 90.0,
            "stages": [
                {
                    "name": f"Stage with {over_value}",
                    "key": "stage_1",
                    "type": "flow",
                    "dynamics": {
                        "points": [[0, 4]],
                        "over": over_value,
                        "interpolation": "linear",
                    },
                    "exit_triggers": [{"type": "time", "value": 30, "relative": True}],
                    "limits": [],
                }
            ],
        }
        is_valid, errors = validator.validate(profile, level=ValidationLevel.MACHINE)
        assert is_valid
        over_errors = [e for e in errors if "dynamics.over" in e.lower()]
        assert len(over_errors) == 0


def test_validate_invalid_stage_type_fails(validator):
    """Test validation fails for invalid stage types."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Invalid Type",
                "key": "stage_1",
                "type": "invalid_type",  # Invalid
                "dynamics": {
                    "points": [[0, 4]],
                    "over": "time",
                    "interpolation": "linear",
                },
                "exit_triggers": [{"type": "time", "value": 30, "relative": True}],
                "limits": [],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    assert not is_valid
    assert any("invalid type" in e.lower() and "invalid_type" in e for e in errors)


def test_validate_valid_stage_types_pass(validator):
    """Test validation passes for valid stage types."""
    for stage_type in ["power", "flow", "pressure"]:
        profile = {
            "name": "Test Profile",
            "id": "test-id",
            "temperature": 90.0,
            "stages": [
                {
                    "name": f"Stage with {stage_type}",
                    "key": "stage_1",
                    "type": stage_type,
                    "dynamics": {
                        "points": [[0, 4]],
                        "over": "time",
                        "interpolation": "linear",
                    },
                    "exit_triggers": [{"type": "time", "value": 30, "relative": True}],
                    "limits": [],
                }
            ],
        }
        is_valid, errors = validator.validate(profile)
        type_errors = [e for e in errors if "invalid type" in e.lower()]
        assert len(type_errors) == 0


def test_validate_invalid_exit_trigger_type_fails(validator):
    """Test validation fails for invalid exit trigger types."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Invalid Trigger",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {
                    "points": [[0, 4]],
                    "over": "time",
                    "interpolation": "linear",
                },
                "exit_triggers": [{"type": "invalid_trigger", "value": 30}],
                "limits": [],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    assert not is_valid
    assert any("exit trigger" in e.lower() and "invalid_trigger" in e for e in errors)


def test_validate_valid_exit_trigger_types_pass(validator):
    """Test validation passes for valid exit trigger types."""
    valid_types = ["weight", "pressure", "flow", "time", "piston_position", "power", "user_interaction"]
    for trigger_type in valid_types:
        profile = {
            "name": "Test Profile",
            "id": "test-id",
            "temperature": 90.0,
            "stages": [
                {
                    "name": f"Stage with {trigger_type}",
                    "key": "stage_1",
                    "type": "flow",
                    "dynamics": {
                        "points": [[0, 4]],
                        "over": "time",
                        "interpolation": "linear",
                    },
                    "exit_triggers": [{"type": trigger_type, "value": 30, "relative": True}],
                    "limits": [],
                }
            ],
        }
        is_valid, errors = validator.validate(profile)
        trigger_errors = [e for e in errors if "exit trigger" in e.lower() and "invalid" in e.lower()]
        assert len(trigger_errors) == 0


def test_validate_invalid_comparison_fails(validator):
    """Test validation fails for invalid comparison operators."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Invalid Comparison",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {
                    "points": [[0, 4]],
                    "over": "time",
                    "interpolation": "linear",
                },
                "exit_triggers": [{"type": "time", "value": 30, "comparison": "=="}],  # Invalid
                "limits": [],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    assert not is_valid
    assert any("comparison" in e.lower() and "==" in e for e in errors)


def test_validate_valid_comparisons_pass(validator):
    """Test validation passes for valid comparison operators."""
    for comparison in [">=", "<="]:
        profile = {
            "name": "Test Profile",
            "id": "test-id",
            "temperature": 90.0,
            "stages": [
                {
                    "name": f"Stage with {comparison}",
                    "key": "stage_1",
                    "type": "flow",
                    "dynamics": {
                        "points": [[0, 4]],
                        "over": "time",
                        "interpolation": "linear",
                    },
                    "exit_triggers": [{"type": "time", "value": 30, "comparison": comparison, "relative": True}],
                    "limits": [],
                }
            ],
        }
        is_valid, errors = validator.validate(profile)
        comparison_errors = [e for e in errors if "comparison" in e.lower() and "invalid" in e.lower()]
        assert len(comparison_errors) == 0


def test_validate_invalid_limit_type_fails(validator):
    """Test validation fails for invalid limit types."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Invalid Limit",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {
                    "points": [[0, 4]],
                    "over": "time",
                    "interpolation": "linear",
                },
                "exit_triggers": [{"type": "time", "value": 30, "relative": True}],
                "limits": [{"type": "invalid_limit", "value": 10}],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    assert not is_valid
    assert any("limit" in e.lower() and "invalid_limit" in e for e in errors)


def test_validate_valid_limit_types_pass(validator):
    """Test validation passes for valid limit types when not redundant with stage type."""
    # Pressure limit on flow stage - valid
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Flow stage with pressure limit",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {
                    "points": [[0, 4]],
                    "over": "time",
                    "interpolation": "linear",
                },
                "exit_triggers": [{"type": "time", "value": 30, "relative": True}],
                "limits": [{"type": "pressure", "value": 10}],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    limit_errors = [e for e in errors if "limit" in e.lower()]
    assert len(limit_errors) == 0

    # Flow limit on pressure stage - valid
    profile2 = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Pressure stage with flow limit",
                "key": "stage_1",
                "type": "pressure",
                "dynamics": {
                    "points": [[0, 9]],
                    "over": "time",
                    "interpolation": "linear",
                },
                "exit_triggers": [{"type": "time", "value": 30, "relative": True}],
                "limits": [{"type": "flow", "value": 4}],
            }
        ],
    }
    is_valid2, errors2 = validator.validate(profile2)
    limit_errors2 = [e for e in errors2 if "limit" in e.lower()]
    assert len(limit_errors2) == 0


def test_validate_redundant_flow_limit_on_flow_stage_fails(validator):
    """Test validation fails when a flow stage has a flow limit (redundant)."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Pre-Wet & Bloom",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {
                    "points": [[0, 3]],
                    "over": "time",
                    "interpolation": "linear",
                },
                "exit_triggers": [{"type": "time", "value": 30, "relative": True}],
                "limits": [{"type": "flow", "value": 3}],  # Redundant!
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    assert not is_valid
    assert any("redundant" in e.lower() and "flow" in e.lower() for e in errors)


def test_validate_redundant_pressure_limit_on_pressure_stage_fails(validator):
    """Test validation fails when a pressure stage has a pressure limit (redundant)."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Fruity Extraction",
                "key": "stage_1",
                "type": "pressure",
                "dynamics": {
                    "points": [[0, 9]],
                    "over": "time",
                    "interpolation": "linear",
                },
                "exit_triggers": [{"type": "time", "value": 30, "relative": True}],
                "limits": [{"type": "pressure", "value": 9}],  # Redundant!
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    assert not is_valid
    assert any("redundant" in e.lower() and "pressure" in e.lower() for e in errors)


# ==================== Exit Trigger Matches Stage Type Tests ====================

def test_validate_flow_stage_with_flow_exit_trigger_fails(validator):
    """Test validation fails when flow stage has flow exit trigger."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Flow Stage",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {
                    "points": [[0, 4]],
                    "over": "time",
                    "interpolation": "linear",
                },
                "exit_triggers": [{"type": "flow", "value": 3, "relative": True}],
                "limits": [{"type": "pressure", "value": 10}],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    assert not is_valid
    assert any("flow" in e.lower() and "control stage" in e.lower() and "exit trigger" in e.lower() for e in errors)


def test_validate_pressure_stage_with_pressure_exit_trigger_fails(validator):
    """Test validation fails when pressure stage has pressure exit trigger."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Pressure Stage",
                "key": "stage_1",
                "type": "pressure",
                "dynamics": {
                    "points": [[0, 9]],
                    "over": "time",
                    "interpolation": "linear",
                },
                "exit_triggers": [{"type": "pressure", "value": 9, "relative": True}],
                "limits": [{"type": "flow", "value": 5}],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    assert not is_valid
    assert any("pressure" in e.lower() and "control stage" in e.lower() and "exit trigger" in e.lower() for e in errors)


def test_validate_flow_stage_with_weight_exit_trigger_passes(validator):
    """Test validation passes when flow stage has weight exit trigger."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Flow Stage",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {
                    "points": [[0, 4]],
                    "over": "time",
                    "interpolation": "linear",
                },
                "exit_triggers": [
                    {"type": "weight", "value": 36, "relative": True},
                    {"type": "time", "value": 45, "relative": True}
                ],
                "limits": [{"type": "pressure", "value": 10}],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    stage_trigger_errors = [e for e in errors if "control stage" in e.lower() and "exit trigger" in e.lower()]
    assert len(stage_trigger_errors) == 0


# ==================== Backup Exit Trigger Tests ====================

def test_validate_single_non_time_trigger_fails(validator):
    """Test validation fails when stage has only one non-time exit trigger."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Flow Stage",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {
                    "points": [[0, 4]],
                    "over": "time",
                    "interpolation": "linear",
                },
                "exit_triggers": [{"type": "weight", "value": 36, "relative": True}],  # Only weight, no time backup
                "limits": [{"type": "pressure", "value": 10}],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    assert not is_valid
    assert any("backup" in e.lower() or "failsafe" in e.lower() for e in errors)


def test_validate_single_time_trigger_passes(validator):
    """Test validation passes when stage has only time trigger (it's a failsafe itself)."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Timed Stage",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {
                    "points": [[0, 4]],
                    "over": "time",
                    "interpolation": "linear",
                },
                "exit_triggers": [{"type": "time", "value": 30, "relative": True}],
                "limits": [{"type": "pressure", "value": 10}],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    backup_errors = [e for e in errors if "backup" in e.lower() or "failsafe" in e.lower()]
    assert len(backup_errors) == 0


def test_validate_multiple_triggers_passes(validator):
    """Test validation passes when stage has multiple exit triggers."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Multi-Trigger Stage",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {
                    "points": [[0, 4]],
                    "over": "time",
                    "interpolation": "linear",
                },
                "exit_triggers": [
                    {"type": "weight", "value": 36, "relative": True},
                    {"type": "time", "value": 45, "relative": True}
                ],
                "limits": [{"type": "pressure", "value": 10}],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    backup_errors = [e for e in errors if "backup" in e.lower() or "failsafe" in e.lower()]
    assert len(backup_errors) == 0


# ==================== Required Limits Tests ====================

def test_validate_flow_stage_without_pressure_limit_fails(validator):
    """Test validation fails when flow stage has no pressure limit."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Flow Stage",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {
                    "points": [[0, 4]],
                    "over": "time",
                    "interpolation": "linear",
                },
                "exit_triggers": [{"type": "time", "value": 30, "relative": True}],
                "limits": [],  # No pressure limit!
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    assert not is_valid
    assert any("flow" in e.lower() and "pressure limit" in e.lower() for e in errors)


def test_validate_pressure_stage_without_flow_limit_fails(validator):
    """Test validation fails when pressure stage has no flow limit."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Pressure Stage",
                "key": "stage_1",
                "type": "pressure",
                "dynamics": {
                    "points": [[0, 9]],
                    "over": "time",
                    "interpolation": "linear",
                },
                "exit_triggers": [{"type": "time", "value": 30, "relative": True}],
                "limits": [],  # No flow limit!
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    assert not is_valid
    assert any("pressure" in e.lower() and "flow limit" in e.lower() for e in errors)


def test_validate_flow_stage_with_pressure_limit_passes(validator):
    """Test validation passes when flow stage has pressure limit."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Flow Stage",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {
                    "points": [[0, 4]],
                    "over": "time",
                    "interpolation": "linear",
                },
                "exit_triggers": [{"type": "time", "value": 30, "relative": True}],
                "limits": [{"type": "pressure", "value": 10}],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    limit_errors = [e for e in errors if "pressure limit" in e.lower() or "flow limit" in e.lower()]
    assert len(limit_errors) == 0


def test_validate_pressure_stage_with_flow_limit_passes(validator):
    """Test validation passes when pressure stage has flow limit."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Pressure Stage",
                "key": "stage_1",
                "type": "pressure",
                "dynamics": {
                    "points": [[0, 9]],
                    "over": "time",
                    "interpolation": "linear",
                },
                "exit_triggers": [{"type": "time", "value": 30, "relative": True}],
                "limits": [{"type": "flow", "value": 5}],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    limit_errors = [e for e in errors if "pressure limit" in e.lower() or "flow limit" in e.lower()]
    assert len(limit_errors) == 0


def test_validate_low_flow_stage_recommends_lower_pressure(validator):
    """Test validation recommends 3 bar for low-flow stages (< 2 ml/s)."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Pre-infusion",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {
                    "points": [[0, 1.5]],  # Low flow — below 2 ml/s threshold
                    "over": "time",
                    "interpolation": "linear",
                },
                "exit_triggers": [{"type": "time", "value": 30, "relative": True}],
                "limits": [],  # Missing pressure limit
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    assert not is_valid
    # Should recommend 3 bar for low-flow stages
    assert any("3 bar" in e and "low-flow" in e for e in errors)


def test_validate_low_pressure_stage_no_flow_limit_passes(validator):
    """Test that low-pressure stages (< 6 bar) don't require a flow limit."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Gentle Pre-wet",
                "key": "stage_1",
                "type": "pressure",
                "dynamics": {
                    "points": [[0, 2], [10, 3]],
                    "over": "time",
                    "interpolation": "linear",
                },
                "exit_triggers": [{"type": "time", "value": 15, "relative": True}],
                "limits": [],  # No flow limit, but low pressure
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    flow_limit_errors = [e for e in errors if "flow limit" in e.lower()]
    assert len(flow_limit_errors) == 0


def test_validate_high_pressure_stage_no_flow_limit_fails(validator):
    """Test that high-pressure stages (>= 6 bar) require a flow limit."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Extraction",
                "key": "stage_1",
                "type": "pressure",
                "dynamics": {
                    "points": [[0, 9]],
                    "over": "time",
                    "interpolation": "linear",
                },
                "exit_triggers": [{"type": "time", "value": 30, "relative": True}],
                "limits": [],  # No flow limit at 9 bar — dangerous
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    assert not is_valid
    assert any("flow limit" in e.lower() for e in errors)


def test_validate_pressure_variable_refs_no_flow_limit_fails(validator):
    """Test that unresolvable variable refs still trigger flow limit warning."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "variables": [
            {"name": "Pressure", "key": "p", "adjustable": True, "value": 9.0}
        ],
        "stages": [
            {
                "name": "Variable Stage",
                "key": "stage_1",
                "type": "pressure",
                "dynamics": {
                    "points": [[0, "$p"]],
                    "over": "time",
                    "interpolation": "linear",
                },
                "exit_triggers": [{"type": "time", "value": 30, "relative": True}],
                "limits": [],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    assert any("flow limit" in e.lower() for e in errors)


def test_advise_returns_suggestions_for_low_value_hold(validator):
    """Test advise() returns suggestions for a low-value hold stage with absolute triggers."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Fill",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {"points": [[0, 1.5]], "over": "time", "interpolation": "linear"},
                "exit_triggers": [{"type": "time", "value": 10, "relative": True}],
                "limits": [{"type": "pressure", "value": 3}],
            },
            {
                "name": "Bloom",
                "key": "stage_2",
                "type": "flow",
                "dynamics": {"points": [[0, 0]], "over": "time", "interpolation": "linear"},
                "exit_triggers": [{"type": "time", "value": 15, "relative": False}],
                "limits": [{"type": "pressure", "value": 3}],
            },
        ],
    }
    suggestions = validator.advise(profile)
    # Should suggest relative triggers for the hold/soak stage
    assert any("hold/soak" in s.lower() for s in suggestions)
    # Should suggest weight trigger for the low-value stage
    assert any("weight" in s.lower() for s in suggestions)


def test_advise_empty_for_well_formed_profile(validator):
    """Test advise() returns empty list for a well-formed profile."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Extraction",
                "key": "stage_1",
                "type": "pressure",
                "dynamics": {"points": [[0, 6], [30, 9]], "over": "time", "interpolation": "linear"},
                "exit_triggers": [
                    {"type": "weight", "value": 36, "relative": True},
                    {"type": "time", "value": 45, "relative": True},
                ],
                "limits": [{"type": "flow", "value": 5}],
            }
        ],
    }
    suggestions = validator.advise(profile)
    assert len(suggestions) == 0


# ==================== Absolute Weight Trigger Tests ====================

def test_validate_absolute_weight_decreasing_fails(validator):
    """Test validation fails when absolute weight trigger decreases across stages."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Saturation",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {"points": [[0, 3]], "over": "time", "interpolation": "linear"},
                "exit_triggers": [
                    {"type": "weight", "value": 5, "relative": False, "comparison": ">="},
                    {"type": "time", "value": 30, "relative": True}
                ],
                "limits": [{"type": "pressure", "value": 3}],
            },
            {
                "name": "Bloom",
                "key": "stage_2",
                "type": "flow",
                "dynamics": {"points": [[0, 0]], "over": "time", "interpolation": "linear"},
                "exit_triggers": [
                    {"type": "weight", "value": 1, "relative": False, "comparison": ">="},  # Lower than previous!
                    {"type": "time", "value": 10, "relative": True}
                ],
                "limits": [{"type": "pressure", "value": 3}],
            },
        ],
    }
    is_valid, errors = validator.validate(profile)
    assert not is_valid
    assert any("absolute weight trigger" in e.lower() and "fire immediately" in e.lower() for e in errors)


def test_validate_absolute_weight_increasing_passes(validator):
    """Test validation passes when absolute weight triggers increase across stages."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Pre-infusion",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {"points": [[0, 3]], "over": "time", "interpolation": "linear"},
                "exit_triggers": [
                    {"type": "weight", "value": 5, "relative": False, "comparison": ">="},
                    {"type": "time", "value": 30, "relative": True}
                ],
                "limits": [{"type": "pressure", "value": 3}],
            },
            {
                "name": "Extraction",
                "key": "stage_2",
                "type": "flow",
                "dynamics": {"points": [[0, 4]], "over": "time", "interpolation": "linear"},
                "exit_triggers": [
                    {"type": "weight", "value": 36, "relative": False, "comparison": ">="},  # Higher - OK
                    {"type": "time", "value": 45, "relative": True}
                ],
                "limits": [{"type": "pressure", "value": 10}],
            },
        ],
    }
    is_valid, errors = validator.validate(profile)
    weight_errors = [e for e in errors if "absolute weight trigger" in e.lower()]
    assert len(weight_errors) == 0


def test_validate_relative_weight_triggers_no_conflict(validator):
    """Test validation passes when using relative weight triggers."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Pre-infusion",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {"points": [[0, 3]], "over": "time", "interpolation": "linear"},
                "exit_triggers": [
                    {"type": "weight", "value": 5, "relative": False, "comparison": ">="},
                    {"type": "time", "value": 30, "relative": True}
                ],
                "limits": [{"type": "pressure", "value": 3}],
            },
            {
                "name": "Bloom",
                "key": "stage_2",
                "type": "flow",
                "dynamics": {"points": [[0, 0]], "over": "time", "interpolation": "linear"},
                "exit_triggers": [
                    {"type": "weight", "value": 1, "relative": True, "comparison": ">="},  # Relative - OK
                    {"type": "time", "value": 10, "relative": True}
                ],
                "limits": [{"type": "pressure", "value": 3}],
            },
        ],
    }
    is_valid, errors = validator.validate(profile)
    weight_errors = [e for e in errors if "absolute weight trigger" in e.lower()]
    assert len(weight_errors) == 0


def test_advise_hold_stage_with_absolute_triggers(validator):
    """Test advise() suggests relative triggers for hold/soak stages."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Fill",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {"points": [[0, 3]], "over": "time", "interpolation": "linear"},
                "exit_triggers": [{"type": "time", "value": 10, "relative": True}],
                "limits": [{"type": "pressure", "value": 3}],
            },
            {
                "name": "The Bloom Room",
                "key": "stage_2",
                "type": "flow",
                "dynamics": {"points": [[0, 0]], "over": "time", "interpolation": "linear"},
                "exit_triggers": [{"type": "time", "value": 15, "relative": False}],  # Absolute
                "limits": [{"type": "pressure", "value": 3}],
            },
        ],
    }
    suggestions = validator.advise(profile)
    assert any("hold/soak" in s.lower() and "relative" in s.lower() for s in suggestions)


def test_lint_low_absolute_weight_in_later_stage_warns(validator):
    """Test linting warns about low absolute weight triggers in non-first stages."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "First Stage",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {"points": [[0, 3]], "over": "time", "interpolation": "linear"},
                "exit_triggers": [{"type": "time", "value": 10, "relative": True}],
                "limits": [{"type": "pressure", "value": 3}],
            },
            {
                "name": "Second Stage",
                "key": "stage_2",
                "type": "flow",
                "dynamics": {"points": [[0, 4]], "over": "time", "interpolation": "linear"},
                "exit_triggers": [
                    {"type": "weight", "value": 5, "relative": False},  # Low absolute in stage 2
                    {"type": "time", "value": 30, "relative": True}
                ],
                "limits": [{"type": "pressure", "value": 10}],
            },
        ],
    }
    warnings = validator.lint(profile)
    assert any("low absolute weight" in w.lower() and "5g" in w for w in warnings)


# ==================== VARIABLES ARRAY REQUIREMENT TESTS ====================


def test_lint_missing_variables_array_warns(validator):
    """Test that missing variables array generates a warning (app compatibility)."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {"points": [[0, 3]], "over": "time"},
                "exit_triggers": [{"type": "time", "value": 30}],
            }
        ],
        # Note: no "variables" key at all
    }
    warnings = validator.lint(profile)
    assert any("missing 'variables' array" in w.lower() for w in warnings)


def test_lint_empty_variables_array_no_warning(validator):
    """Test that empty variables array does NOT generate a warning (legitimate choice)."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "variables": [],  # Empty array is a valid design choice
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {"points": [[0, 3]], "over": "time"},
                "exit_triggers": [{"type": "time", "value": 30}],
            }
        ],
    }
    warnings = validator.lint(profile)
    assert not any("no variables defined" in w.lower() for w in warnings)
    assert not any("missing 'variables' array" in w.lower() for w in warnings)


def test_lint_profile_with_variables_no_warning(validator):
    """Test that profile with proper variables doesn't warn about missing variables."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "variables": [
            {"name": "Target Pressure", "key": "target_pressure", "type": "pressure", "value": 8.0}
        ],
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "pressure",
                "dynamics": {"points": [[0, "$target_pressure"]], "over": "time"},
                "exit_triggers": [{"type": "time", "value": 30}],
            }
        ],
    }
    warnings = validator.lint(profile)
    # Should not have any variable-related warnings
    assert not any("no variables defined" in w.lower() for w in warnings)
    assert not any("missing 'variables' array" in w.lower() for w in warnings)


def test_lint_does_not_warn_unused_variable(validator):
    """Test that lint() does NOT warn about unused variables (handled by validate at SAFETY level)."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "variables": [
            {"name": "Used Var", "key": "active_pressure", "type": "pressure", "value": 8.0},
            {"name": "Unused Var", "key": "dormant_flow", "type": "flow", "value": 3.0},
        ],
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "pressure",
                "dynamics": {"points": [[0, "$active_pressure"]], "over": "time"},
                "exit_triggers": [{"type": "time", "value": 30}],
            }
        ],
    }
    warnings = validator.lint(profile)
    # lint() should NOT flag unused variables — that's validate()'s job
    assert not any("dormant_flow" in w.lower() and "never used" in w.lower() for w in warnings)


# Tests for _validate_variables (validation errors, not lint warnings)

def test_validate_variables_info_without_emoji_fails(validator):
    """Test that info variables without emoji prefix fail STRICT validation."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "variables": [
            {"name": "Target Weight", "key": "target_weight", "adjustable": False, "value": 36}
        ],
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "pressure",
                "dynamics": {"points": [[0, 8]], "over": "time"},
                "exit_triggers": [{"type": "time", "value": 30}],
                "limits": [{"type": "flow", "value": 5}],
            }
        ],
    }
    is_valid, errors = validator.validate(profile, level=ValidationLevel.STRICT)
    assert not is_valid
    assert any("info variable" in e.lower() and "emoji prefix" in e.lower() for e in errors)


def test_validate_variables_info_with_emoji_passes(validator):
    """Test that info variables with emoji prefix pass STRICT validation."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "variables": [
            {"name": "ℹ️ Target Weight", "key": "target_weight", "adjustable": False, "value": 36}
        ],
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "pressure",
                "dynamics": {"points": [[0, 8]], "over": "time"},
                "exit_triggers": [{"type": "time", "value": 30}],
                "limits": [{"type": "flow", "value": 5}],
            }
        ],
    }
    is_valid, errors = validator.validate(profile, level=ValidationLevel.STRICT)
    # Should not have emoji-related errors
    assert not any("emoji" in e.lower() for e in errors)


def test_validate_variables_adjustable_with_emoji_fails(validator):
    """Test that adjustable variables with emoji prefix fail STRICT validation."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "variables": [
            {"name": "🎯 Target Pressure", "key": "target_pressure", "adjustable": True, "value": 8.0}
        ],
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "pressure",
                "dynamics": {"points": [[0, "$target_pressure"]], "over": "time"},
                "exit_triggers": [{"type": "time", "value": 30}],
                "limits": [{"type": "flow", "value": 5}],
            }
        ],
    }
    is_valid, errors = validator.validate(profile, level=ValidationLevel.STRICT)
    assert not is_valid
    assert any("adjustable variable" in e.lower() and "should not have an emoji" in e.lower() for e in errors)


def test_validate_variables_adjustable_without_emoji_passes(validator):
    """Test that adjustable variables without emoji prefix pass STRICT validation."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "variables": [
            {"name": "Target Pressure", "key": "target_pressure", "adjustable": True, "value": 8.0}
        ],
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "pressure",
                "dynamics": {"points": [[0, "$target_pressure"]], "over": "time"},
                "exit_triggers": [{"type": "time", "value": 30}],
                "limits": [{"type": "flow", "value": 5}],
            }
        ],
    }
    is_valid, errors = validator.validate(profile, level=ValidationLevel.STRICT)
    # Should not have emoji-related errors
    assert not any("emoji" in e.lower() for e in errors)


def test_validate_variables_unused_adjustable_fails(validator):
    """Test that unused adjustable variables fail validation."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "variables": [
            {"name": "Unused Pressure", "key": "unused_pressure", "adjustable": True, "value": 8.0}
        ],
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "pressure",
                "dynamics": {"points": [[0, 8]], "over": "time"},  # Not using $unused_pressure
                "exit_triggers": [{"type": "time", "value": 30}],
                "limits": [{"type": "flow", "value": 5}],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    assert not is_valid
    assert any("unused_pressure" in e.lower() and "never used" in e.lower() for e in errors)


def test_validate_variables_unused_info_passes(validator):
    """Test that unused info variables pass validation (they're display-only)."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "variables": [
            {"name": "ℹ️ Roast Level", "key": "roast_level", "adjustable": False, "value": "Medium"}
        ],
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "pressure",
                "dynamics": {"points": [[0, 8]], "over": "time"},
                "exit_triggers": [{"type": "time", "value": 30}],
                "limits": [{"type": "flow", "value": 5}],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    # Unused info vars should not cause errors (they're for display)
    assert not any("roast_level" in e.lower() and "never used" in e.lower() for e in errors)


def test_validate_variable_used_in_limit_is_not_unused(validator):
    """Test that a variable referenced in a limit value is counted as used."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "variables": [
            {"name": "Max Flow", "key": "max_flow", "adjustable": True, "value": 4.0}
        ],
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "pressure",
                "dynamics": {"points": [[0, 8]], "over": "time"},
                "exit_triggers": [{"type": "time", "value": 30}],
                "limits": [{"type": "flow", "value": "$max_flow"}],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    assert not any("max_flow" in e.lower() and "never used" in e.lower() for e in errors)


def test_validate_variable_used_in_exit_trigger_is_not_unused(validator):
    """Test that a variable referenced in an exit trigger value is counted as used."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "variables": [
            {"name": "Target Weight", "key": "target_weight", "adjustable": True, "value": 40.0}
        ],
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "pressure",
                "dynamics": {"points": [[0, 8]], "over": "time"},
                "exit_triggers": [
                    {"type": "weight", "value": "$target_weight"},
                    {"type": "time", "value": 60},
                ],
                "limits": [{"type": "flow", "value": 5}],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    assert not any("target_weight" in e.lower() and "never used" in e.lower() for e in errors)


def test_validate_variable_used_only_in_dynamics_is_not_unused(validator):
    """Test that a variable referenced only in dynamics points is counted as used."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "variables": [
            {"name": "Target Pressure", "key": "target_pressure", "adjustable": True, "value": 8.0}
        ],
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "pressure",
                "dynamics": {"points": [[0, "$target_pressure"]], "over": "time"},
                "exit_triggers": [{"type": "time", "value": 30}],
                "limits": [{"type": "flow", "value": 5}],
            }
        ],
    }
    is_valid, errors = validator.validate(profile)
    assert not any("target_pressure" in e.lower() and "never used" in e.lower() for e in errors)


# ==================== ValidationLevel System Tests ====================


# Shared helper profile that has ONLY a redundant limit (flow limit on flow stage)
# and is otherwise valid at the MACHINE level (no pressure issues, good types, etc.).
def _make_redundant_limit_profile():
    return {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "variables": [],
        "stages": [
            {
                "name": "Pre-Wet",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {"points": [[0, 3], [10, 3]], "over": "time", "interpolation": "linear"},
                "exit_triggers": [{"type": "time", "value": 30, "relative": True}],
                "limits": [{"type": "flow", "value": 3}],  # Redundant: flow on flow stage
            }
        ],
    }


def _make_no_backup_trigger_profile():
    """Profile with only a weight exit trigger — no time-based failsafe."""
    return {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "variables": [],
        "stages": [
            {
                "name": "Extraction",
                "key": "stage_1",
                "type": "flow",
                "dynamics": {"points": [[0, 4], [10, 4]], "over": "time", "interpolation": "linear"},
                "exit_triggers": [{"type": "weight", "value": 36, "relative": True}],
                "limits": [{"type": "pressure", "value": 9}],
            }
        ],
    }


def _make_emoji_info_variable_profile():
    """Profile with an info variable that is MISSING the required emoji prefix."""
    return {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "variables": [
            {"name": "Roast Level", "key": "roast_level", "adjustable": False, "value": "Medium"}
        ],
        "stages": [
            {
                "name": "Stage 1",
                "key": "stage_1",
                "type": "pressure",
                "dynamics": {"points": [[0, 9], [10, 9]], "over": "time", "interpolation": "linear"},
                "exit_triggers": [{"type": "time", "value": 30, "relative": True}],
                "limits": [{"type": "flow", "value": 5}],
            }
        ],
    }


def test_validate_default_is_safety_level(validator):
    """Test that the default validation level is SAFETY."""
    profile = _make_redundant_limit_profile()

    # Default call — should behave the same as explicitly passing SAFETY
    is_valid_default, errors_default = validator.validate(profile)
    is_valid_safety, errors_safety = validator.validate(profile, level=ValidationLevel.SAFETY)

    assert is_valid_default == is_valid_safety
    assert errors_default == errors_safety


def test_validate_machine_level_skips_redundant_limits(validator):
    """Test that redundant limits are NOT caught at MACHINE level."""
    profile = _make_redundant_limit_profile()
    is_valid, errors = validator.validate(profile, level=ValidationLevel.MACHINE)
    redundant_errors = [e for e in errors if "redundant" in e.lower()]
    assert len(redundant_errors) == 0


def test_validate_safety_level_catches_redundant_limits(validator):
    """Test that redundant limits ARE caught at SAFETY (default) level."""
    profile = _make_redundant_limit_profile()
    is_valid, errors = validator.validate(profile, level=ValidationLevel.SAFETY)
    assert not is_valid
    assert any("redundant" in e.lower() and "flow" in e.lower() for e in errors)


def test_validate_machine_level_skips_backup_trigger_check(validator):
    """Test that the missing-backup-trigger check is skipped at MACHINE level."""
    profile = _make_no_backup_trigger_profile()
    is_valid, errors = validator.validate(profile, level=ValidationLevel.MACHINE)
    backup_errors = [e for e in errors if "backup" in e.lower() or "failsafe" in e.lower()]
    assert len(backup_errors) == 0


def test_validate_safety_level_catches_backup_trigger(validator):
    """Test that a missing backup trigger is caught at SAFETY level."""
    profile = _make_no_backup_trigger_profile()
    is_valid, errors = validator.validate(profile, level=ValidationLevel.SAFETY)
    assert not is_valid
    assert any("backup" in e.lower() or "failsafe" in e.lower() for e in errors)


def test_validate_safety_level_skips_emoji_rules(validator):
    """Test that emoji naming rules are NOT enforced at SAFETY (default) level."""
    profile = _make_emoji_info_variable_profile()
    is_valid, errors = validator.validate(profile, level=ValidationLevel.SAFETY)
    emoji_errors = [e for e in errors if "emoji" in e.lower()]
    assert len(emoji_errors) == 0


def test_validate_strict_level_catches_emoji_rules(validator):
    """Test that emoji naming rules ARE enforced at STRICT level."""
    profile = _make_emoji_info_variable_profile()
    is_valid, errors = validator.validate(profile, level=ValidationLevel.STRICT)
    assert not is_valid
    assert any("info variable" in e.lower() and "emoji prefix" in e.lower() for e in errors)


def test_validate_strict_level_also_runs_safety_checks(validator):
    """Test that STRICT level still runs all SAFETY-level checks."""
    profile = _make_redundant_limit_profile()
    is_valid, errors = validator.validate(profile, level=ValidationLevel.STRICT)
    # Redundant limit should still be caught even at STRICT level
    assert not is_valid
    assert any("redundant" in e.lower() for e in errors)


def test_validate_machine_level_catches_invalid_pressure(validator):
    """Test that over-pressure errors are always caught, even at MACHINE level."""
    profile = {
        "name": "Test Profile",
        "id": "test-id",
        "temperature": 90.0,
        "stages": [
            {
                "name": "High Pressure Stage",
                "key": "stage_1",
                "type": "pressure",
                "dynamics": {"points": [[0, 20]], "over": "time", "interpolation": "linear"},
                "exit_triggers": [{"type": "time", "value": 30, "relative": True}],
                "limits": [{"type": "flow", "value": 5}],
            }
        ],
    }
    is_valid, errors = validator.validate(profile, level=ValidationLevel.MACHINE)
    assert not is_valid
    assert any("15 bar limit" in e.lower() for e in errors)


def test_validate_level_passed_to_validate_and_raise(validator):
    """Test that validate_and_raise respects the level parameter."""
    profile = _make_emoji_info_variable_profile()

    # Should NOT raise at SAFETY level (emoji rules not checked)
    validator.validate_and_raise(profile, level=ValidationLevel.SAFETY)

    # Should raise at STRICT level (emoji rules enforced)
    with pytest.raises(ProfileValidationError) as exc_info:
        validator.validate_and_raise(profile, level=ValidationLevel.STRICT)
    assert any("emoji" in e.lower() for e in exc_info.value.errors)


def test_validator_constructor_default_level(sample_schema):
    """Test that the constructor default level can be overridden."""
    # Create a validator with MACHINE as default
    machine_validator = ProfileValidator(schema_path=sample_schema, level=ValidationLevel.MACHINE)

    profile = _make_redundant_limit_profile()
    is_valid, errors = machine_validator.validate(profile)  # No level arg — uses constructor default
    redundant_errors = [e for e in errors if "redundant" in e.lower()]
    assert len(redundant_errors) == 0  # MACHINE default skips redundancy check


def test_validator_constructor_strict_level(sample_schema):
    """Test that a validator created with STRICT default enforces emoji rules without passing level."""
    strict_validator = ProfileValidator(schema_path=sample_schema, level=ValidationLevel.STRICT)
    profile = _make_emoji_info_variable_profile()
    is_valid, errors = strict_validator.validate(profile)  # No level arg — uses STRICT default
    assert not is_valid
    assert any("emoji" in e.lower() for e in errors)
