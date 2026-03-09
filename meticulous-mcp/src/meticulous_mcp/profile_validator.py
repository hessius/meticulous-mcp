"""Profile validator using JSON schema.

Copyright (C) 2024 Meticulous MCP

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import json
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonschema
from jsonschema import ValidationError


class ValidationLevel(Enum):
    """Controls which checks are run during profile validation.

    Levels:
    - MACHINE: Schema conformance + hardware limits only. Use when you trust the
      profile author and only want to catch machine-breaking errors.
    - SAFETY (default): Everything in MACHINE plus shot-safety best-practice
      checks (backup triggers, required limits, redundant limits, etc.).
    - STRICT: Everything in SAFETY plus app-UI convention checks (variable
      naming conventions, emoji rules, etc.).
    """
    MACHINE = "machine"
    SAFETY = "safety"
    STRICT = "strict"


class ProfileValidationError(Exception):
    """Raised when profile validation fails."""

    def __init__(self, message: str, errors: Optional[List[str]] = None):
        """Initialize validation error.
        
        Args:
            message: Error message
            errors: List of detailed validation errors
        """
        self.message = message
        self.errors = errors or []
        
        # Include all errors in the exception message so they're visible when raised
        if self.errors:
            error_lines = [message, ""]
            for i, error in enumerate(self.errors, 1):
                error_lines.append(f"{i}. {error}")
            full_message = "\n".join(error_lines)
        else:
            full_message = message
        
        super().__init__(full_message)


class ProfileValidator:
    """Validates espresso profiles against JSON schema."""

    def __init__(self, schema_path: Optional[str] = None, level: ValidationLevel = ValidationLevel.SAFETY):
        """Initialize the validator.

        Args:
            schema_path: Path to schema.json file. If not provided, attempts to find
                        it relative to this file or in espresso-profile-schema repo.
            level: Default validation level to use when none is passed to validate().
                   Defaults to ValidationLevel.SAFETY.
        """
        self._default_level = level
        possible_paths = []
        if schema_path is None:
            # Try to find schema relative to this file
            current_dir = Path(__file__).parent.parent.parent
            possible_paths = [
                current_dir / "espresso-profile-schema" / "schema.json",
                Path(__file__).parent / "schema.json",
            ]
            for path in possible_paths:
                if path.exists():
                    schema_path = str(path)
                    break
        
        if schema_path is None or not os.path.exists(schema_path):
            paths_str = ", ".join(str(p) for p in possible_paths) if possible_paths else "none"
            raise FileNotFoundError(
                f"Schema file not found. Path given: {schema_path}. Tried: {paths_str}. "
                "Please provide schema_path or ensure espresso-profile-schema is available."
            )

        with open(schema_path, "r", encoding="utf-8") as f:
            self._schema = json.load(f)
        
        # Create validator instance
        self._validator = jsonschema.Draft7Validator(self._schema)

    def validate(self, profile: Dict[str, Any], level: Optional[ValidationLevel] = None) -> Tuple[bool, List[str]]:
        """Validate a profile against the schema.

        Args:
            profile: Profile dictionary to validate
            level: Validation level to use. Defaults to the level set at construction
                   time (ValidationLevel.SAFETY if not specified).

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        if level is None:
            level = self._default_level

        errors = []
        try:
            self._validator.validate(profile)
        except ValidationError as e:
            errors.append(self._format_error(e))

            # Collect all validation errors
            for error in self._validator.iter_errors(profile):
                if error != e:  # Don't duplicate the first error
                    errors.append(self._format_error(error))

        # --- MACHINE level checks (always run) ---------------------------------
        # Hardware limits and schema-level constraints that, if violated, will
        # cause the machine to behave dangerously or reject the profile entirely.

        # Pressure limits (15 bar max)
        errors.extend(self._validate_pressure_limits(profile))

        # Interpolation values (only 'linear' and 'curve' supported)
        errors.extend(self._validate_interpolation(profile))

        # dynamics.over values
        errors.extend(self._validate_dynamics_over(profile))

        # Stage types
        errors.extend(self._validate_stage_types(profile))

        # Exit trigger types and comparison operators
        errors.extend(self._validate_exit_triggers(profile))

        # Limit types (validity only — redundancy is a SAFETY check)
        errors.extend(self._validate_limit_types(profile))

        # Absolute weight trigger monotonicity
        errors.extend(self._validate_absolute_weight_triggers(profile))

        # --- SAFETY level checks (run at SAFETY and STRICT) --------------------
        # Best-practice shot-safety patterns. Violations won't necessarily crash
        # the machine but can cause dangerous shots or indefinite stalls.
        if level in (ValidationLevel.SAFETY, ValidationLevel.STRICT):
            # Redundant limits (same type as stage control)
            errors.extend(self._validate_redundant_limits(profile))

            # Exit trigger type must not match stage control type
            errors.extend(self._validate_exit_trigger_matches_stage_type(profile))

            # Every stage needs a backup/failsafe exit trigger
            errors.extend(self._validate_backup_exit_triggers(profile))

            # Flow stages need pressure limits; pressure stages need flow limits
            errors.extend(self._validate_required_limits(profile))

            # Adjustable variables that are defined but never used
            errors.extend(self._validate_unused_adjustable_variables(profile))

        # --- STRICT level checks (run only at STRICT) --------------------------
        # App-UI convention checks. Violations affect the user experience in the
        # Meticulous app but do not affect shot safety or machine operation.
        if level == ValidationLevel.STRICT:
            # Variable naming conventions (emoji rules)
            errors.extend(self._validate_variable_naming(profile))

        return len(errors) == 0, errors

    def validate_and_raise(self, profile: Dict[str, Any], level: Optional[ValidationLevel] = None) -> None:
        """Validate a profile and raise ProfileValidationError if invalid.

        Args:
            profile: Profile dictionary to validate
            level: Validation level to use. Defaults to the level set at construction
                   time (ValidationLevel.SAFETY if not specified).

        Raises:
            ProfileValidationError: If validation fails (includes all errors in message)
        """
        is_valid, errors = self.validate(profile, level=level)
        if not is_valid:
            message = f"Profile validation failed with {len(errors)} error(s)"
            # The ProfileValidationError will automatically include all errors in its message
            raise ProfileValidationError(message, errors)

    def _validate_pressure_limits(self, profile: Dict[str, Any]) -> List[str]:
        """Validate pressure limits (15 bar max) in profile.
        
        Args:
            profile: Profile dictionary to validate
            
        Returns:
            List of pressure-related validation errors
        """
        errors = []
        
        if "stages" not in profile or not isinstance(profile["stages"], list):
            return errors
        
        for i, stage in enumerate(profile["stages"]):
            if not isinstance(stage, dict):
                continue
            
            stage_name = stage.get("name", f"Stage {i+1}")
            
            # Check pressure in dynamics points (only for pressure-type stages)
            if stage.get("type") == "pressure":
                dynamics = stage.get("dynamics", {})
                points = dynamics.get("points", [])
                for point_idx, point in enumerate(points):
                    if isinstance(point, list) and len(point) >= 2:
                        pressure_val = point[1]
                        if isinstance(pressure_val, (int, float)):
                            if pressure_val > 15:
                                errors.append(f"Stage '{stage_name}' dynamics point {point_idx+1} has pressure {pressure_val} bar which exceeds the 15 bar limit. Please reduce pressure to 15 bar or below.")
                            elif pressure_val < 0:
                                errors.append(f"Stage '{stage_name}' dynamics point {point_idx+1} has negative pressure {pressure_val} bar. Pressure must be non-negative.")
            
            # Check pressure in exit triggers
            exit_triggers = stage.get("exit_triggers", [])
            for trigger_idx, trigger in enumerate(exit_triggers):
                if isinstance(trigger, dict) and trigger.get("type") == "pressure":
                    pressure_val = trigger.get("value")
                    if isinstance(pressure_val, (int, float)):
                        if pressure_val > 15:
                            errors.append(f"Stage '{stage_name}' exit trigger {trigger_idx+1} has pressure {pressure_val} bar which exceeds the 15 bar limit. Please reduce pressure to 15 bar or below.")
                        elif pressure_val < 0:
                            errors.append(f"Stage '{stage_name}' exit trigger {trigger_idx+1} has negative pressure {pressure_val} bar. Pressure must be non-negative.")
        
        return errors

    def _validate_interpolation(self, profile: Dict[str, Any]) -> List[str]:
        """Validate interpolation values in profile dynamics.
        
        The Meticulous machine only supports 'linear' and 'curve' interpolation.
        The value 'none' is not supported and will cause the machine to stall.
        
        Args:
            profile: Profile dictionary to validate
            
        Returns:
            List of interpolation-related validation errors
        """
        errors = []
        valid_interpolations = {"linear", "curve"}
        
        if "stages" not in profile or not isinstance(profile["stages"], list):
            return errors
        
        for i, stage in enumerate(profile["stages"]):
            if not isinstance(stage, dict):
                continue
            
            stage_name = stage.get("name", f"Stage {i+1}")
            dynamics = stage.get("dynamics", {})
            
            if isinstance(dynamics, dict):
                interpolation = dynamics.get("interpolation")
                if interpolation is not None and interpolation not in valid_interpolations:
                    errors.append(
                        f"Stage '{stage_name}' has invalid interpolation value '{interpolation}'. "
                        f"Only 'linear' and 'curve' are supported. "
                        f"The value 'none' is not supported by the Meticulous machine and will cause it to stall."
                    )
                
                # Check that 'curve' interpolation has at least 2 points
                points = dynamics.get("points", [])
                if interpolation == "curve" and len(points) < 2:
                    errors.append(
                        f"Stage '{stage_name}' uses 'curve' interpolation but has only {len(points)} point(s). "
                        f"Curve interpolation requires at least 2 points. Use 'linear' for single-point dynamics."
                    )
        
        return errors

    def _validate_dynamics_over(self, profile: Dict[str, Any]) -> List[str]:
        """Validate dynamics.over values in profile.
        
        The 'over' field must be one of: 'time', 'weight', 'piston_position'.
        
        Args:
            profile: Profile dictionary to validate
            
        Returns:
            List of dynamics.over validation errors
        """
        errors = []
        valid_over_values = {"time", "weight", "piston_position"}
        
        if "stages" not in profile or not isinstance(profile["stages"], list):
            return errors
        
        for i, stage in enumerate(profile["stages"]):
            if not isinstance(stage, dict):
                continue
            
            stage_name = stage.get("name", f"Stage {i+1}")
            dynamics = stage.get("dynamics", {})
            
            if isinstance(dynamics, dict):
                over = dynamics.get("over")
                if over is not None and over not in valid_over_values:
                    errors.append(
                        f"Stage '{stage_name}' has invalid dynamics.over value '{over}'. "
                        f"Must be one of: 'time', 'weight', 'piston_position'."
                    )
        
        return errors

    def _validate_stage_types(self, profile: Dict[str, Any]) -> List[str]:
        """Validate stage type values in profile.
        
        Stage type must be one of: 'power', 'flow', 'pressure'.
        
        Args:
            profile: Profile dictionary to validate
            
        Returns:
            List of stage type validation errors
        """
        errors = []
        valid_stage_types = {"power", "flow", "pressure"}
        
        if "stages" not in profile or not isinstance(profile["stages"], list):
            return errors
        
        for i, stage in enumerate(profile["stages"]):
            if not isinstance(stage, dict):
                continue
            
            stage_name = stage.get("name", f"Stage {i+1}")
            stage_type = stage.get("type")
            
            if stage_type is not None and stage_type not in valid_stage_types:
                errors.append(
                    f"Stage '{stage_name}' has invalid type '{stage_type}'. "
                    f"Must be one of: 'power', 'flow', 'pressure'."
                )
        
        return errors

    def _validate_exit_triggers(self, profile: Dict[str, Any]) -> List[str]:
        """Validate exit trigger values in profile.
        
        Exit trigger type must be one of: 'weight', 'pressure', 'flow', 'time', 
        'piston_position', 'power', 'user_interaction'.
        Comparison must be one of: '>=', '<='.
        
        Args:
            profile: Profile dictionary to validate
            
        Returns:
            List of exit trigger validation errors
        """
        errors = []
        valid_trigger_types = {"weight", "pressure", "flow", "time", "piston_position", "power", "user_interaction"}
        valid_comparisons = {">=", "<="}
        
        if "stages" not in profile or not isinstance(profile["stages"], list):
            return errors
        
        for i, stage in enumerate(profile["stages"]):
            if not isinstance(stage, dict):
                continue
            
            stage_name = stage.get("name", f"Stage {i+1}")
            exit_triggers = stage.get("exit_triggers", [])
            
            for trigger_idx, trigger in enumerate(exit_triggers):
                if not isinstance(trigger, dict):
                    continue
                
                trigger_type = trigger.get("type")
                if trigger_type is not None and trigger_type not in valid_trigger_types:
                    errors.append(
                        f"Stage '{stage_name}' exit trigger {trigger_idx+1} has invalid type '{trigger_type}'. "
                        f"Must be one of: {', '.join(sorted(valid_trigger_types))}."
                    )
                
                comparison = trigger.get("comparison")
                if comparison is not None and comparison not in valid_comparisons:
                    errors.append(
                        f"Stage '{stage_name}' exit trigger {trigger_idx+1} has invalid comparison '{comparison}'. "
                        f"Must be one of: '>=', '<='."
                    )
        
        return errors

    def _validate_limit_types(self, profile: Dict[str, Any]) -> List[str]:
        """Validate limit type values in profile (MACHINE level check).

        Limit type must be one of: 'pressure', 'flow'.

        Args:
            profile: Profile dictionary to validate

        Returns:
            List of limit type validation errors
        """
        errors = []
        valid_limit_types = {"pressure", "flow"}

        if "stages" not in profile or not isinstance(profile["stages"], list):
            return errors

        for i, stage in enumerate(profile["stages"]):
            if not isinstance(stage, dict):
                continue

            stage_name = stage.get("name", f"Stage {i+1}")
            limits = stage.get("limits", [])

            if not isinstance(limits, list):
                continue

            for limit_idx, limit in enumerate(limits):
                if not isinstance(limit, dict):
                    continue

                limit_type = limit.get("type")
                if limit_type is not None and limit_type not in valid_limit_types:
                    errors.append(
                        f"Stage '{stage_name}' limit {limit_idx+1} has invalid type '{limit_type}'. "
                        f"Must be one of: 'pressure', 'flow'."
                    )

        return errors

    def _validate_redundant_limits(self, profile: Dict[str, Any]) -> List[str]:
        """Validate that limits are not redundant with stage control type (SAFETY level check).

        A limit cannot have the same type as the stage control type, as this is
        redundant and the Meticulous app will reject it.

        Args:
            profile: Profile dictionary to validate

        Returns:
            List of redundant limit validation errors
        """
        errors = []

        if "stages" not in profile or not isinstance(profile["stages"], list):
            return errors

        for i, stage in enumerate(profile["stages"]):
            if not isinstance(stage, dict):
                continue

            stage_name = stage.get("name", f"Stage {i+1}")
            stage_type = stage.get("type")
            limits = stage.get("limits", [])

            if not isinstance(limits, list):
                continue

            for limit in limits:
                if not isinstance(limit, dict):
                    continue

                limit_type = limit.get("type")
                if limit_type is not None and stage_type is not None and limit_type == stage_type:
                    errors.append(
                        f"Stage '{stage_name}' has a '{limit_type}' limit but is a '{stage_type}' control stage. "
                        f"This is redundant - you cannot limit {limit_type} when you're already controlling {stage_type}. "
                        f"Use a '{('pressure' if limit_type == 'flow' else 'flow')}' limit instead, or remove the limit."
                    )

        return errors

    def _validate_exit_trigger_matches_stage_type(self, profile: Dict[str, Any]) -> List[str]:
        """Validate that exit trigger types don't match the stage control type.
        
        If a stage controls flow, it should not have a flow exit trigger as the primary trigger.
        If a stage controls pressure, it should not have a pressure exit trigger.
        This creates a paradox where grind variations mean the trigger may never fire.
        
        Args:
            profile: Profile dictionary to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        if "stages" not in profile or not isinstance(profile["stages"], list):
            return errors
        
        for i, stage in enumerate(profile["stages"]):
            if not isinstance(stage, dict):
                continue
            
            stage_name = stage.get("name", f"Stage {i+1}")
            stage_type = stage.get("type")
            exit_triggers = stage.get("exit_triggers", [])
            
            if not stage_type or not exit_triggers:
                continue
            
            # Check if any exit trigger has the same type as the stage control
            for trigger_idx, trigger in enumerate(exit_triggers):
                if not isinstance(trigger, dict):
                    continue
                
                trigger_type = trigger.get("type")
                
                # Check for matching types (flow stage with flow trigger, pressure stage with pressure trigger)
                if trigger_type == stage_type and stage_type in ("flow", "pressure"):
                    errors.append(
                        f"Stage '{stage_name}' is a '{stage_type}' control stage but has a '{trigger_type}' exit trigger. "
                        f"This is problematic - if you're controlling {stage_type}, you can't reliably exit based on {trigger_type} "
                        f"since it's what you're already controlling. Use a different trigger type like 'time', 'weight', or "
                        f"'{('pressure' if stage_type == 'flow' else 'flow')}'."
                    )
        
        return errors

    def _validate_backup_exit_triggers(self, profile: Dict[str, Any]) -> List[str]:
        """Validate that every stage has a backup/failsafe exit trigger.
        
        Every stage should have either:
        - Multiple exit triggers (OR logic provides failsafe)
        - At least one time-based trigger (universal failsafe)
        
        This prevents shots from stalling indefinitely if grind is wrong.
        
        Args:
            profile: Profile dictionary to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        if "stages" not in profile or not isinstance(profile["stages"], list):
            return errors
        
        for i, stage in enumerate(profile["stages"]):
            if not isinstance(stage, dict):
                continue
            
            stage_name = stage.get("name", f"Stage {i+1}")
            exit_triggers = stage.get("exit_triggers", [])
            
            if not exit_triggers:
                # Already caught by other validation
                continue
            
            # Count valid triggers and check for time trigger
            valid_triggers = [t for t in exit_triggers if isinstance(t, dict) and t.get("type")]
            has_time_trigger = any(t.get("type") == "time" for t in valid_triggers)
            
            # Must have either multiple triggers OR a time trigger
            if len(valid_triggers) == 1 and not has_time_trigger:
                trigger_type = valid_triggers[0].get("type", "unknown")
                errors.append(
                    f"Stage '{stage_name}' has only one exit trigger ('{trigger_type}') with no time-based failsafe. "
                    f"Every stage must have a backup exit condition to prevent indefinite stalls. "
                    f"Add a time-based trigger (e.g., 'time >= 45s') as a failsafe, or add a second trigger like 'weight'."
                )
        
        return errors

    def _validate_required_limits(self, profile: Dict[str, Any]) -> List[str]:
        """Validate that every stage has required safety limits.

        Flow stages must have a pressure limit to prevent machine stall at high pressure.
        Pressure stages at >= 6 bar must have a flow limit to prevent gusher shots.
        Low-pressure stages (< 6 bar) are not a gusher risk and are skipped.

        Recommended limit values are based on stage dynamics rather than stage names.

        Args:
            profile: Profile dictionary to validate

        Returns:
            List of validation errors
        """
        errors = []

        if "stages" not in profile or not isinstance(profile["stages"], list):
            return errors

        for i, stage in enumerate(profile["stages"]):
            if not isinstance(stage, dict):
                continue

            stage_name = stage.get("name", f"Stage {i+1}")
            stage_type = stage.get("type")
            limits = stage.get("limits", [])

            if not stage_type:
                continue

            # Get limit types present
            limit_types = set()
            if isinstance(limits, list):
                for limit in limits:
                    if isinstance(limit, dict) and limit.get("type"):
                        limit_types.add(limit.get("type"))

            is_low_value = self._is_low_value_stage(stage)

            # Flow stages need pressure limits
            if stage_type == "flow" and "pressure" not in limit_types:
                recommended_limit = "3 bar" if is_low_value else "10 bar"
                errors.append(
                    f"Stage '{stage_name}' is a 'flow' control stage but has no pressure limit. "
                    f"This is dangerous - if the grind is too fine, pressure could spike to 12+ bar and stall. "
                    f"Add a pressure limit (recommended: {recommended_limit} for {'low-flow' if is_low_value else 'extraction'} stages)."
                )

            # Pressure stages need flow limits — only when target pressure is >= 6 bar
            # or when the target is unresolvable (variable references).
            if stage_type == "pressure" and "flow" not in limit_types:
                max_pressure = self._get_max_dynamics_value(stage)
                if max_pressure is None or max_pressure >= 6.0:
                    errors.append(
                        f"Stage '{stage_name}' is a 'pressure' control stage but has no flow limit. "
                        f"This can cause gusher shots if the grind is too coarse. "
                        f"Add a flow limit (recommended: 5 ml/s)."
                    )

        return errors

    def _validate_absolute_weight_triggers(self, profile: Dict[str, Any]) -> List[str]:
        """Validate that absolute weight triggers are strictly increasing across stages.
        
        If Stage N exits at absolute weight X, and Stage N+1 has absolute weight trigger Y,
        then Y must be > X, otherwise the trigger will fire immediately.
        
        Args:
            profile: Profile dictionary to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        if "stages" not in profile or not isinstance(profile["stages"], list):
            return errors
        
        stages = profile["stages"]
        
        # Track the maximum absolute weight trigger seen so far
        max_absolute_weight = 0.0
        max_weight_stage_name = None
        
        for i, stage in enumerate(stages):
            if not isinstance(stage, dict):
                continue
            
            stage_name = stage.get("name", f"Stage {i+1}")
            exit_triggers = stage.get("exit_triggers", [])
            
            # Find absolute weight triggers in this stage
            for trigger in exit_triggers:
                if not isinstance(trigger, dict):
                    continue
                
                trigger_type = trigger.get("type")
                trigger_value = trigger.get("value")
                is_relative = trigger.get("relative", False)
                
                if trigger_type == "weight" and not is_relative and isinstance(trigger_value, (int, float)):
                    # This is an absolute weight trigger
                    if trigger_value <= max_absolute_weight and max_weight_stage_name is not None:
                        errors.append(
                            f"Stage '{stage_name}' has absolute weight trigger ({trigger_value}g) that is <= "
                            f"the previous stage '{max_weight_stage_name}' weight trigger ({max_absolute_weight}g). "
                            f"This trigger will fire immediately since the scale already shows >= {max_absolute_weight}g. "
                            f"Use 'relative: true' for stage-specific weight tracking, or increase the weight threshold."
                        )
                    
                    # Update max for next stages
                    if trigger_value > max_absolute_weight:
                        max_absolute_weight = trigger_value
                        max_weight_stage_name = stage_name
        
        return errors

    def _build_emoji_pattern(self) -> "re.Pattern[str]":
        """Build a compiled regex for detecting an emoji at the start of a string."""
        return re.compile(
            r'^['
            r'\U0001F300-\U0001F9FF'  # Miscellaneous Symbols and Pictographs, Emoticons, etc.
            r'\U00002600-\U000027BF'  # Misc symbols, Dingbats
            r'\U00002100-\U0000214F'  # Letterlike Symbols (includes ℹ️ U+2139)
            r'\U0001F000-\U0001F02F'  # Mahjong tiles
            r'\U0001FA00-\U0001FAFF'  # Extended-A symbols
            r'\uFE00-\uFE0F'          # Variation Selectors (emoji presentation)
            r']'
        )

    def _variable_usage_map(self, profile: Dict[str, Any]) -> Dict[str, bool]:
        """Return a dict mapping variable key → True if used in any stage dynamics."""
        used: Dict[str, bool] = {}
        if "stages" not in profile:
            return used
        for stage in profile["stages"]:
            if not isinstance(stage, dict):
                continue
            dynamics = stage.get("dynamics", {})
            points = dynamics.get("points", [])
            for point in points:
                if isinstance(point, list):
                    for val in point:
                        if isinstance(val, str) and val.startswith("$"):
                            used[val[1:]] = True
        return used

    def _get_max_dynamics_value(self, stage: Dict[str, Any]) -> Optional[float]:
        """Return the maximum numeric target value from a stage's dynamics points.

        Variable references ($key) are skipped — we can only classify stages
        with concrete numeric values.  Returns None when no numeric target
        values are found.
        """
        dynamics = stage.get("dynamics", {})
        points = dynamics.get("points", [])
        max_val: Optional[float] = None
        for point in points:
            if isinstance(point, list):
                for val in point[1:]:  # skip the x-axis value at index 0
                    if isinstance(val, (int, float)):
                        if max_val is None or val > max_val:
                            max_val = val
        return max_val

    def _is_low_value_stage(self, stage: Dict[str, Any]) -> bool:
        """Determine if a stage targets low values (pre-infusion-like behavior).

        Classification thresholds:
        - Flow stages: max target < 2 ml/s
        - Pressure stages: max target < 4 bar

        Returns False if dynamics contain only variable references (unresolvable).
        """
        stage_type = stage.get("type")
        max_val = self._get_max_dynamics_value(stage)
        if max_val is None:
            return False
        if stage_type == "flow":
            return max_val < 2.0
        if stage_type == "pressure":
            return max_val < 4.0
        return False

    def _is_hold_stage(self, stage: Dict[str, Any]) -> bool:
        """Determine if a stage holds constant (bloom/soak-like behavior).

        A hold stage has either a single dynamics point or all numeric target
        values are identical (flat curve).  Only evaluates numeric values.
        """
        dynamics = stage.get("dynamics", {})
        points = dynamics.get("points", [])
        numeric_values = []
        for point in points:
            if isinstance(point, list) and len(point) >= 2:
                val = point[1]
                if isinstance(val, (int, float)):
                    numeric_values.append(val)
        if len(numeric_values) <= 1:
            return True
        return all(v == numeric_values[0] for v in numeric_values)

    def _validate_variable_naming(self, profile: Dict[str, Any]) -> List[str]:
        """Validate variable naming conventions — emoji rules (STRICT level check).

        Rules:
        1. Info variables (adjustable=False) MUST have an emoji prefix in their name.
        2. Adjustable variables (adjustable=True) must NOT have an emoji prefix.

        Args:
            profile: Profile dictionary to validate

        Returns:
            List of variable naming validation errors
        """
        errors = []
        variables = profile.get("variables", [])
        if not variables:
            return errors

        emoji_pattern = self._build_emoji_pattern()

        for var in variables:
            if not isinstance(var, dict):
                continue
            key = var.get("key")
            name = var.get("name", "")
            is_info = not var.get("adjustable", True)

            if not key:
                continue

            has_emoji = bool(emoji_pattern.match(name))

            if is_info and not has_emoji:
                errors.append(
                    f"Info variable '{key}' ({name}) must have an emoji prefix in its name. "
                    f"Info variables are displayed to help the user understand the profile but "
                    f"cannot be adjusted. Add an emoji like ℹ️, 📊, or 💡 at the start."
                )

            if not is_info and has_emoji:
                errors.append(
                    f"Adjustable variable '{key}' ({name}) should not have an emoji prefix. "
                    f"Emoji prefixes are reserved for info (non-adjustable) variables to "
                    f"visually distinguish them in the UI."
                )

        return errors

    def _validate_unused_adjustable_variables(self, profile: Dict[str, Any]) -> List[str]:
        """Validate that all adjustable variables are used in stage dynamics (SAFETY level check).

        Adjustable variables are intended to be referenced as $key in dynamics points.
        An adjustable variable that is never used is most likely a mistake.

        Info variables (adjustable=False) are display-only and do not need to be used.

        Args:
            profile: Profile dictionary to validate

        Returns:
            List of unused-variable validation errors
        """
        errors = []
        variables = profile.get("variables", [])
        if not variables:
            return errors

        used_map = self._variable_usage_map(profile)

        for var in variables:
            if not isinstance(var, dict):
                continue
            key = var.get("key")
            name = var.get("name", "")
            is_info = not var.get("adjustable", True)

            if key and not is_info and not used_map.get(key, False):
                errors.append(
                    f"Adjustable variable '{key}' ({name}) is defined but never used in any "
                    f"stage dynamics. Either use it with ${key} in a dynamics point, mark it as "
                    f"info-only (adjustable: false), or remove it."
                )

        return errors

    def _format_error(self, error: ValidationError) -> str:
        """Format a validation error into a readable message.
        
        Args:
            error: The ValidationError instance
            
        Returns:
            Formatted error message with helpful context
        """
        path = " -> ".join(str(p) for p in error.path)
        message = error.message
        
        # Add helpful context for common errors
        if "required" in message.lower() or "missing" in message.lower():
            if "stages" in path.lower():
                message += " (each stage must have: name, key, type, dynamics, exit_triggers)"
            elif "dynamics" in path.lower():
                message += " (dynamics must have: points, over, interpolation)"
            elif "exit_triggers" in path.lower():
                message += " (each exit trigger must have: type, value)"
        
        if "Field required" in message:
            # Extract field name from path
            field_name = path.split(" -> ")[-1] if path else "unknown"
            message = f"Missing required field '{field_name}'"
        
        if path:
            return f"Field '{path}': {message}"
        return f"Root level: {message}"

    def lint(self, profile: Dict[str, Any]) -> List[str]:
        """Lint a profile and return warnings/suggestions.
        
        Args:
            profile: Profile dictionary to lint
            
        Returns:
            List of linting warnings/suggestions
        """
        warnings = []
        
        # Check for common issues
        if "stages" in profile:
            stages = profile["stages"]
            if not isinstance(stages, list):
                warnings.append("'stages' should be a list")
            elif len(stages) == 0:
                warnings.append("Profile has no stages")
            else:
                # Check stage ordering and naming
                for i, stage in enumerate(stages):
                    if not isinstance(stage, dict):
                        continue
                    
                    stage_name = stage.get("name", f"Stage {i+1}")
                    stage_key = stage.get("key", f"stage_{i+1}")
                    
                    # Check exit triggers
                    exit_triggers = stage.get("exit_triggers", [])
                    if not exit_triggers:
                        warnings.append(f"Stage '{stage_name}' has no exit triggers - stages should have at least one exit trigger")
                    else:
                        # Check exit trigger types
                        has_weight_trigger = any(et.get("type") == "weight" for et in exit_triggers if isinstance(et, dict))
                        has_time_trigger = any(et.get("type") == "time" for et in exit_triggers if isinstance(et, dict))
                        if not has_weight_trigger and not has_time_trigger:
                            warnings.append(f"Stage '{stage_name}' has exit triggers but none are weight or time-based - consider adding a weight/time trigger")
                        
                        # Check for missing 'relative' field in exit triggers
                        # The machine requires 'relative' to always be present (defaults to false)
                        for idx, trigger in enumerate(exit_triggers):
                            if isinstance(trigger, dict):
                                if "relative" not in trigger or trigger.get("relative") is None:
                                    warnings.append(f"Stage '{stage_name}' exit trigger {idx+1} ({trigger.get('type', 'unknown')}) is missing 'relative' field - will be normalized to false. The machine requires 'relative' to always be present in exit triggers.")
                    
                    # Check dynamics
                    dynamics = stage.get("dynamics")
                    if dynamics:
                        points = dynamics.get("points", [])
                        if not points:
                            warnings.append(f"Stage '{stage_name}' has empty dynamics points - dynamics should define pressure/flow changes")
                        elif len(points) == 1:
                            warnings.append(f"Stage '{stage_name}' has only one dynamics point - consider adding more points for smoother transitions")
                        
                        over = dynamics.get("over", "")
                        if over not in ["time", "weight", "piston_position"]:
                            warnings.append(f"Stage '{stage_name}' has invalid dynamics.over value '{over}' - should be 'time', 'weight', or 'piston_position'")
                        
                        # Check interpolation value
                        interpolation = dynamics.get("interpolation", "")
                        if interpolation not in ["linear", "curve"]:
                            warnings.append(f"Stage '{stage_name}' has invalid interpolation '{interpolation}' - should be 'linear' or 'curve'. The value 'none' is not supported.")
                    
                    # Check stage type
                    stage_type = stage.get("type", "")
                    if stage_type not in ["power", "flow", "pressure"]:
                        warnings.append(f"Stage '{stage_name}' has invalid type '{stage_type}' - should be 'power', 'flow', or 'pressure'")
                    
                    # Check for missing or None 'limits' field
                    # The machine requires 'limits' to always be present as an array (even if empty)
                    if "limits" not in stage:
                        warnings.append(f"Stage '{stage_name}' is missing 'limits' field - will be normalized to empty array []. The machine requires 'limits' to always be present as an array in stages.")
                    elif stage.get("limits") is None:
                        warnings.append(f"Stage '{stage_name}' has 'limits' set to null - will be normalized to empty array []. The machine requires 'limits' to always be an array, not null.")
                    
                    # Check for duplicate keys
                    if i > 0:
                        prev_keys = [s.get("key") for s in stages[:i] if isinstance(s, dict)]
                        if stage_key in prev_keys:
                            warnings.append(f"Stage '{stage_name}' has duplicate key '{stage_key}' - stage keys should be unique")
                    
                    # Check limit values for sensible bounds
                    limits = stage.get("limits", [])
                    if isinstance(limits, list):
                        for limit in limits:
                            if not isinstance(limit, dict):
                                continue
                            limit_type = limit.get("type")
                            limit_value = limit.get("value")
                            if isinstance(limit_value, (int, float)):
                                if limit_type == "pressure":
                                    if limit_value < 0:
                                        warnings.append(f"Stage '{stage_name}' has negative pressure limit ({limit_value} bar) - should be >= 0")
                                    elif limit_value > 12:
                                        warnings.append(f"Stage '{stage_name}' has very high pressure limit ({limit_value} bar) - consider lowering to 10-12 bar for safety")
                                elif limit_type == "flow":
                                    if limit_value < 0:
                                        warnings.append(f"Stage '{stage_name}' has negative flow limit ({limit_value} ml/s) - should be >= 0")
                                    elif limit_value > 8:
                                        warnings.append(f"Stage '{stage_name}' has very high flow limit ({limit_value} ml/s) - consider lowering to 5-6 ml/s for better control")

                    # Check for low absolute weight triggers in non-first stages
                    if i > 0:  # Not the first stage
                        for trigger in exit_triggers:
                            if isinstance(trigger, dict):
                                trigger_type = trigger.get("type")
                                trigger_value = trigger.get("value")
                                is_relative = trigger.get("relative", False)
                                if trigger_type == "weight" and not is_relative and isinstance(trigger_value, (int, float)):
                                    if trigger_value < 10:
                                        warnings.append(
                                            f"Stage '{stage_name}' (stage {i+1}) has a low absolute weight trigger ({trigger_value}g). "
                                            f"If preceding stages have weight-based exits, this may fire immediately. "
                                            f"Consider using 'relative: true' for stage-specific weight tracking."
                                        )
        
        # Check temperature range
        if "temperature" in profile:
            temp = profile["temperature"]
            if isinstance(temp, (int, float)):
                if temp < 80 or temp > 100:
                    warnings.append(f"Temperature {temp}°C is outside typical range (80-100°C) - consider adjusting for your roast level")
                elif temp < 85:
                    warnings.append(f"Temperature {temp}°C is on the lower end - suitable for dark roasts")
                elif temp > 95:
                    warnings.append(f"Temperature {temp}°C is on the higher end - suitable for light roasts")

        # Check final_weight
        if "final_weight" in profile:
            weight = profile["final_weight"]
            if isinstance(weight, (int, float)):
                if weight < 10 or weight > 100:
                    warnings.append(f"Final weight {weight}g is outside typical range (10-100g) - verify this is intentional")
                elif weight < 20:
                    warnings.append(f"Final weight {weight}g is quite low - typical espresso shots are 25-45g")
                elif weight > 60:
                    warnings.append(f"Final weight {weight}g is quite high - this approaches lungo/ristretto territory")
        
        # Check variables - the variables array should always exist for app compatibility
        if "variables" not in profile:
            warnings.append(
                "Profile is missing 'variables' array - this field must be present (even if empty) "
                "for Meticulous app compatibility. The app may crash when trying to add variables."
            )
        else:
            variables = profile.get("variables", [])
            if variables:
                # Check for undefined variable references in stages
                var_keys = [v.get("key") for v in variables if isinstance(v, dict)]

                if "stages" in profile:
                    stages = profile["stages"]
                    for stage in stages:
                        if not isinstance(stage, dict):
                            continue
                        dynamics = stage.get("dynamics", {})
                        points = dynamics.get("points", [])
                        for point in points:
                            if isinstance(point, list) and len(point) >= 2:
                                for val in point:
                                    if isinstance(val, str) and val.startswith("$"):
                                        var_key = val[1:]  # Remove $
                                        if var_key not in var_keys:
                                            warnings.append(f"Stage '{stage.get('name', 'unknown')}' references variable '${var_key}' but it's not defined in variables")

        return warnings

    def advise(self, profile: Dict[str, Any]) -> List[str]:
        """Return opt-in design guidance for a profile.

        Unlike lint() which flags structural anomalies, advise() suggests
        design improvements based on espresso best practices.  These are
        subjective recommendations, not errors.

        Args:
            profile: Profile dictionary to analyze

        Returns:
            List of design suggestions
        """
        suggestions: List[str] = []

        if "stages" not in profile or not isinstance(profile["stages"], list):
            return suggestions

        stages = profile["stages"]
        for i, stage in enumerate(stages):
            if not isinstance(stage, dict):
                continue

            stage_name = stage.get("name", f"Stage {i+1}")
            exit_triggers = stage.get("exit_triggers", [])
            limits = stage.get("limits", [])
            is_low_value = self._is_low_value_stage(stage)
            is_hold = self._is_hold_stage(stage)

            # Suggestion: Low-value stages with high pressure limits
            if is_low_value and isinstance(limits, list):
                for limit in limits:
                    if isinstance(limit, dict) and limit.get("type") == "pressure":
                        limit_value = limit.get("value")
                        if isinstance(limit_value, (int, float)) and limit_value > 4:
                            suggestions.append(
                                f"Stage '{stage_name}' targets low values but has a pressure limit of "
                                f"{limit_value} bar - consider lowering to 3-4 bar for gentler saturation."
                            )

            # Suggestion: Low-value stages without weight-based exit
            if is_low_value:
                has_weight_trigger = any(
                    et.get("type") == "weight" for et in exit_triggers if isinstance(et, dict)
                )
                if not has_weight_trigger:
                    suggestions.append(
                        f"Stage '{stage_name}' targets low values without a weight-based exit trigger. "
                        f"Consider adding 'weight >= 3-5g' to detect early dripping and adapt to grind variations."
                    )

            # Suggestion: Hold stages at low values with absolute triggers (bloom/soak pattern)
            if is_hold and is_low_value and i > 0:
                has_absolute_trigger = any(
                    isinstance(t, dict) and not t.get("relative", False)
                    for t in exit_triggers
                )
                if has_absolute_trigger:
                    suggestions.append(
                        f"Stage '{stage_name}' appears to be a hold/soak stage but uses absolute exit triggers. "
                        f"Consider using 'relative: true' for exit triggers so the stage duration is "
                        f"independent of previous stages."
                    )

        return suggestions

