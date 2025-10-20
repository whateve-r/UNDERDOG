"""
Mamdani Fuzzy Inference System
Transforms ML outputs and indicators into confidence scores using fuzzy logic rules.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple
from pathlib import Path
import yaml
import numpy as np


# ========================================
# Membership Functions
# ========================================

class MembershipFunction:
    """Base class for membership functions"""
    
    def evaluate(self, x: float) -> float:
        """Evaluate membership degree at point x"""
        raise NotImplementedError


class TriangularMF(MembershipFunction):
    """Triangular membership function"""
    
    def __init__(self, a: float, b: float, c: float):
        """
        Args:
            a: Left point (membership = 0)
            b: Peak point (membership = 1)
            c: Right point (membership = 0)
        """
        self.a = a
        self.b = b
        self.c = c
    
    def evaluate(self, x: float) -> float:
        if x <= self.a or x >= self.c:
            return 0.0
        elif x == self.b:
            return 1.0
        elif x < self.b:
            return (x - self.a) / (self.b - self.a)
        else:
            return (self.c - x) / (self.c - self.b)


class TrapezoidalMF(MembershipFunction):
    """Trapezoidal membership function"""
    
    def __init__(self, a: float, b: float, c: float, d: float):
        """
        Args:
            a: Left point (membership = 0)
            b: Left peak point (membership = 1)
            c: Right peak point (membership = 1)
            d: Right point (membership = 0)
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    
    def evaluate(self, x: float) -> float:
        if x <= self.a or x >= self.d:
            return 0.0
        elif self.b <= x <= self.c:
            return 1.0
        elif x < self.b:
            return (x - self.a) / (self.b - self.a)
        else:
            return (self.d - x) / (self.d - self.c)


class GaussianMF(MembershipFunction):
    """Gaussian membership function"""
    
    def __init__(self, mean: float, sigma: float):
        """
        Args:
            mean: Center of the Gaussian
            sigma: Standard deviation
        """
        self.mean = mean
        self.sigma = sigma
    
    def evaluate(self, x: float) -> float:
        return np.exp(-((x - self.mean) ** 2) / (2 * self.sigma ** 2))


# ========================================
# Fuzzy Variables and Sets
# ========================================

@dataclass
class FuzzySet:
    """Represents a fuzzy set with linguistic label"""
    label: str
    mf: MembershipFunction
    
    def membership(self, x: float) -> float:
        """Calculate membership degree"""
        return self.mf.evaluate(x)


class FuzzyVariable:
    """Represents a fuzzy variable with multiple fuzzy sets"""
    
    def __init__(self, name: str, range_min: float, range_max: float):
        """
        Args:
            name: Variable name
            range_min: Minimum value of universe
            range_max: Maximum value of universe
        """
        self.name = name
        self.range_min = range_min
        self.range_max = range_max
        self.sets: Dict[str, FuzzySet] = {}
    
    def add_set(self, label: str, mf: MembershipFunction):
        """Add a fuzzy set to this variable"""
        self.sets[label] = FuzzySet(label=label, mf=mf)
    
    def fuzzify(self, value: float) -> Dict[str, float]:
        """
        Fuzzify a crisp value into fuzzy memberships.
        
        Args:
            value: Crisp input value
        
        Returns:
            Dict mapping linguistic labels to membership degrees
        """
        # Clip to range
        value = np.clip(value, self.range_min, self.range_max)
        
        return {label: fset.membership(value) for label, fset in self.sets.items()}


# ========================================
# Fuzzy Rules
# ========================================

@dataclass
class FuzzyRule:
    """Represents a single fuzzy rule"""
    antecedents: List[Tuple[str, str]]  # [(var_name, set_label), ...]
    consequent: Tuple[str, str]  # (var_name, set_label)
    weight: float = 1.0
    operator: str = "AND"  # AND or OR
    
    def evaluate(self, fuzzified_inputs: Dict[str, Dict[str, float]]) -> float:
        """
        Evaluate rule firing strength.
        
        Args:
            fuzzified_inputs: Dict of {var_name: {set_label: membership}}
        
        Returns:
            Rule firing strength [0-1]
        """
        memberships = []
        
        for var_name, set_label in self.antecedents:
            if var_name in fuzzified_inputs and set_label in fuzzified_inputs[var_name]:
                memberships.append(fuzzified_inputs[var_name][set_label])
            else:
                memberships.append(0.0)
        
        if not memberships:
            return 0.0
        
        # Apply operator
        if self.operator == "AND":
            firing_strength = min(memberships)  # T-norm: minimum
        else:  # OR
            firing_strength = max(memberships)  # S-norm: maximum
        
        return firing_strength * self.weight


# ========================================
# Mamdani Inference Engine
# ========================================

class MamdaniInference:
    """
    Mamdani fuzzy inference system for confidence scoring.
    """
    
    def __init__(self):
        """Initialize inference engine"""
        self.input_variables: Dict[str, FuzzyVariable] = {}
        self.output_variables: Dict[str, FuzzyVariable] = {}
        self.rules: List[FuzzyRule] = []
    
    def add_input_variable(self, var: FuzzyVariable):
        """Add input fuzzy variable"""
        self.input_variables[var.name] = var
    
    def add_output_variable(self, var: FuzzyVariable):
        """Add output fuzzy variable"""
        self.output_variables[var.name] = var
    
    def add_rule(self, rule: FuzzyRule):
        """Add fuzzy rule"""
        self.rules.append(rule)
    
    def infer(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """
        Perform Mamdani inference.
        
        Args:
            inputs: Dict of {var_name: crisp_value}
        
        Returns:
            Dict of {output_var_name: defuzzified_value}
        """
        # Step 1: Fuzzification
        fuzzified_inputs = {}
        for var_name, value in inputs.items():
            if var_name in self.input_variables:
                fuzzified_inputs[var_name] = self.input_variables[var_name].fuzzify(value)
        
        # Step 2: Rule evaluation and aggregation
        output_aggregations = {name: {} for name in self.output_variables.keys()}
        
        for rule in self.rules:
            firing_strength = rule.evaluate(fuzzified_inputs)
            
            if firing_strength > 0:
                out_var_name, out_set_label = rule.consequent
                
                if out_var_name in output_aggregations:
                    if out_set_label not in output_aggregations[out_var_name]:
                        output_aggregations[out_var_name][out_set_label] = []
                    output_aggregations[out_var_name][out_set_label].append(firing_strength)
        
        # Step 3: Defuzzification (Centroid method)
        results = {}
        for out_var_name, aggregation in output_aggregations.items():
            if out_var_name in self.output_variables:
                results[out_var_name] = self._defuzzify(
                    self.output_variables[out_var_name],
                    aggregation
                )
        
        return results
    
    def _defuzzify(self, output_var: FuzzyVariable, aggregation: Dict[str, List[float]]) -> float:
        """
        Defuzzify using centroid (center of gravity) method.
        
        Args:
            output_var: Output fuzzy variable
            aggregation: Aggregated fuzzy set activations
        
        Returns:
            Crisp output value
        """
        # Discretize the universe
        resolution = 100
        x_range = np.linspace(output_var.range_min, output_var.range_max, resolution)
        
        # Calculate aggregated membership for each point
        membership_values = np.zeros(resolution)
        
        for set_label, firing_strengths in aggregation.items():
            if set_label in output_var.sets:
                # Use maximum of all firing strengths for this set
                max_firing = max(firing_strengths)
                
                # Calculate clipped membership
                for i, x in enumerate(x_range):
                    set_membership = output_var.sets[set_label].membership(x)
                    clipped = min(set_membership, max_firing)
                    membership_values[i] = max(membership_values[i], clipped)
        
        # Centroid calculation
        numerator = np.sum(x_range * membership_values)
        denominator = np.sum(membership_values)
        
        if denominator == 0:
            # No activation, return midpoint
            return (output_var.range_min + output_var.range_max) / 2
        
        return numerator / denominator
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'MamdaniInference':
        """
        Load fuzzy inference system from YAML configuration.
        
        Args:
            yaml_path: Path to YAML config file
        
        Returns:
            Configured MamdaniInference instance
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        system = cls()
        
        # Load input variables
        for var_config in config.get('input_variables', []):
            var = FuzzyVariable(
                name=var_config['name'],
                range_min=var_config['range'][0],
                range_max=var_config['range'][1]
            )
            
            # Add fuzzy sets
            for set_config in var_config['sets']:
                mf = cls._create_membership_function(set_config['mf'])
                var.add_set(set_config['label'], mf)
            
            system.add_input_variable(var)
        
        # Load output variables
        for var_config in config.get('output_variables', []):
            var = FuzzyVariable(
                name=var_config['name'],
                range_min=var_config['range'][0],
                range_max=var_config['range'][1]
            )
            
            for set_config in var_config['sets']:
                mf = cls._create_membership_function(set_config['mf'])
                var.add_set(set_config['label'], mf)
            
            system.add_output_variable(var)
        
        # Load rules
        for rule_config in config.get('rules', []):
            rule = FuzzyRule(
                antecedents=[(a['var'], a['set']) for a in rule_config['if']],
                consequent=(rule_config['then']['var'], rule_config['then']['set']),
                weight=rule_config.get('weight', 1.0),
                operator=rule_config.get('operator', 'AND')
            )
            system.add_rule(rule)
        
        return system
    
    @staticmethod
    def _create_membership_function(mf_config: Dict) -> MembershipFunction:
        """Create membership function from config"""
        mf_type = mf_config['type']
        
        if mf_type == 'triangular':
            return TriangularMF(mf_config['a'], mf_config['b'], mf_config['c'])
        elif mf_type == 'trapezoidal':
            return TrapezoidalMF(mf_config['a'], mf_config['b'], mf_config['c'], mf_config['d'])
        elif mf_type == 'gaussian':
            return GaussianMF(mf_config['mean'], mf_config['sigma'])
        else:
            raise ValueError(f"Unknown membership function type: {mf_type}")


# ========================================
# Confidence Scoring System
# ========================================

class ConfidenceScorer:
    """
    High-level confidence scoring system using fuzzy logic.
    Transforms ML model outputs and indicators into actionable confidence scores.
    """
    
    def __init__(self, rules_path: Optional[str] = None):
        """
        Initialize confidence scorer.
        
        Args:
            rules_path: Path to fuzzy rules YAML file
        """
        if rules_path and Path(rules_path).exists():
            self.fuzzy_system = MamdaniInference.from_yaml(rules_path)
        else:
            self.fuzzy_system = self._create_default_system()
    
    def score(self,
             ml_probability: float,
             atr_ratio: float,
             momentum: float,
             volume_ratio: Optional[float] = None) -> float:
        """
        Calculate confidence score from inputs.
        
        Args:
            ml_probability: ML model probability [0-1]
            atr_ratio: Current ATR / Historical ATR
            momentum: Momentum indicator (normalized)
            volume_ratio: Volume / Average volume
        
        Returns:
            Confidence score [0-1]
        """
        inputs = {
            'ml_prob': ml_probability,
            'atr_ratio': atr_ratio,
            'momentum': momentum
        }
        
        if volume_ratio is not None:
            inputs['volume_ratio'] = volume_ratio
        
        results = self.fuzzy_system.infer(inputs)
        return results.get('confidence', 0.5)
    
    def _create_default_system(self) -> MamdaniInference:
        """Create default fuzzy inference system"""
        system = MamdaniInference()
        
        # Input: ML Probability
        ml_prob = FuzzyVariable('ml_prob', 0.0, 1.0)
        ml_prob.add_set('low', TrapezoidalMF(0.0, 0.0, 0.4, 0.5))
        ml_prob.add_set('medium', TriangularMF(0.4, 0.5, 0.6))
        ml_prob.add_set('high', TrapezoidalMF(0.5, 0.6, 1.0, 1.0))
        system.add_input_variable(ml_prob)
        
        # Input: ATR Ratio
        atr_ratio = FuzzyVariable('atr_ratio', 0.0, 3.0)
        atr_ratio.add_set('low', TrapezoidalMF(0.0, 0.0, 0.5, 1.0))
        atr_ratio.add_set('normal', TriangularMF(0.8, 1.0, 1.2))
        atr_ratio.add_set('high', TrapezoidalMF(1.0, 1.5, 3.0, 3.0))
        system.add_input_variable(atr_ratio)
        
        # Input: Momentum
        momentum = FuzzyVariable('momentum', -1.0, 1.0)
        momentum.add_set('negative', TrapezoidalMF(-1.0, -1.0, -0.3, 0.0))
        momentum.add_set('neutral', TriangularMF(-0.2, 0.0, 0.2))
        momentum.add_set('positive', TrapezoidalMF(0.0, 0.3, 1.0, 1.0))
        system.add_input_variable(momentum)
        
        # Output: Confidence
        confidence = FuzzyVariable('confidence', 0.0, 1.0)
        confidence.add_set('very_low', TrapezoidalMF(0.0, 0.0, 0.2, 0.3))
        confidence.add_set('low', TriangularMF(0.2, 0.35, 0.5))
        confidence.add_set('medium', TriangularMF(0.4, 0.5, 0.6))
        confidence.add_set('high', TriangularMF(0.5, 0.65, 0.8))
        confidence.add_set('very_high', TrapezoidalMF(0.7, 0.8, 1.0, 1.0))
        system.add_output_variable(confidence)
        
        # Rules
        system.add_rule(FuzzyRule(
            [('ml_prob', 'high'), ('atr_ratio', 'normal'), ('momentum', 'positive')],
            ('confidence', 'very_high')
        ))
        system.add_rule(FuzzyRule(
            [('ml_prob', 'high'), ('atr_ratio', 'normal')],
            ('confidence', 'high')
        ))
        system.add_rule(FuzzyRule(
            [('ml_prob', 'medium'), ('momentum', 'positive')],
            ('confidence', 'medium')
        ))
        system.add_rule(FuzzyRule(
            [('ml_prob', 'low')],
            ('confidence', 'very_low')
        ))
        system.add_rule(FuzzyRule(
            [('atr_ratio', 'high')],
            ('confidence', 'low')
        ))
        
        return system
