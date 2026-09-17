"""Comprehensive tests for NeatModule save/load with state_dict_with_metadata()."""

import sys
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import ModifiedNEAT as neat
from ModifiedNEAT.nn.base import NeatModule, NeatParameter, PseudoModule
from ModifiedNEAT.config import Config


def pretty_print(data, indent=2):
    def simplify(obj):
        # Handle numpy arrays and torch tensors
        if hasattr(obj, 'shape'):
            return f"<shape={tuple(obj.shape)}, dtype={getattr(obj, 'dtype', '?')}>"
        elif isinstance(obj, dict):
            return {k: simplify(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [simplify(v) for v in obj]
        return obj

    print(json.dumps(simplify(data), indent=indent, default=str))

class SimpleNeatModule(NeatModule):
    """Simple test NeatModule with basic parameters."""
    def __init__(self, input_size=10, output_size=5):
        super().__init__(input_size=input_size, output_size=output_size)
        self.input_size = input_size
        self.output_size = output_size
        
        # Create NeatParameters
        self.weight = NeatParameter((output_size, input_size))
        self.bias = NeatParameter((output_size,))
        
    def forward(self, x):
        return x @ self.weight.t() + self.bias


class NestedNeatModule(NeatModule):
    """Nested NeatModule with sub-modules."""
    def __init__(self, input_size=10, output_size=8, *modules):
        super(NestedNeatModule, self).__init__()
        
        self.input_size  = input_size
        self.output_size = output_size
        self.hidden_size = 64
        
        self.fc1 = SimpleNeatModule(input_size, self.hidden_size)
        self.fc2 = SimpleNeatModule(self.hidden_size, output_size)
        self.actv = nn.Tanh()
        
        for i, module in enumerate(modules):
            setattr(self, f"ex{i+1}", module)
        
    def forward(self, x):
        x = self.actv(self.fc1(x))
        x = self.fc2(x)
        return x


def test_state_dict_with_metadata():
    """Test state_dict_with_metadata() serialization."""
    print("\nTesting state_dict_with_metadata()...")
    
    module = SimpleNeatModule(10, 5)
    print(module)
    
    # Get serialization
    serialized = module.neat_dict()
    
    # Verify structure
    assert 'version' in serialized, "Missing version"
    assert serialized['version'] == 1, "Wrong version"
    assert 'state_dict' in serialized, "Missing state_dict"
    assert 'architecture' in serialized, "Missing architecture"
    assert 'population' in serialized, "Missing population"
    
    # Verify state_dict contains parameters
    assert len(serialized['state_dict']) > 0, "Empty state_dict"    # Should have 'weight' and 'bias' as keys, not numeric indices
    assert 'weight' in serialized['state_dict'], "'weight' not in state_dict"
    assert 'bias' in serialized['state_dict'], "'bias' not in state_dict" 
       
    # Verify architecture contains metadata
    arch = serialized['architecture']
    assert 'neat_params' in arch, "Missing neat_params in architecture"
    assert 'neat_modules' in arch, "Missing neat_modules in architecture"
    
    # Verify population
    pop = serialized['population']
    assert 'genomes_total' in pop, "Missing genome_num in population"
    assert 'mapping' in pop, "Missing mapping in population"
    assert 'params' in pop, "Missing params in population"
    
    print("✓ state_dict_with_metadata() test passed")


def test_strict_load():
    """Test strict loading into same module instance."""
    print("\nTesting strict load into same module...")
    
    # Create and modify original module
    module1 = SimpleNeatModule(10, 5)
    original_weight = module1.weight.data.clone()
    
    # Modify weights
    with torch.no_grad():
        module1.weight[:] = torch.ones_like(module1.weight.data)
        module1.bias[:] = torch.ones_like(module1.bias.data)
    
    # Serialize
    serialized = module1.neat_dict()
    
    # Create new module and load strictly
    module2 = SimpleNeatModule(10, 5)
    loaded = module2.load_neat_dict(
        serialized, 
        strict=True
    )
    
    # Verify parameters match (NeatParameter adds batch dimension)
    assert torch.allclose(loaded.weight.data, torch.ones(1, 5, 10)), "Weights not restored"
    assert torch.allclose(loaded.bias.data, torch.ones(1, 5)), "Bias not restored"
    
    print("✓ Strict load test passed")


def test_strict_init_load():
    """Test strict loading into same module instance."""
    print("\nTesting strict load into same module but initialized...")
    
    # Create and modify original module
    module1 = SimpleNeatModule(10, 5)
    neat.Population(50, module1, Config(), verbose=False)
    
    # Modify weights
    with torch.no_grad():
        module1.weight[:] = torch.ones_like(module1.weight.data)
        module1.bias[:] = torch.ones_like(module1.bias.data)
    
    # Serialize
    serialized = module1.neat_dict()
    
    # Create new module and load strictly
    module2 = SimpleNeatModule(10, 5)
    loaded = module2.load_neat_dict(
        serialized, 
        strict=True
    )
    
    # Verify parameters match (NeatParameter adds batch dimension)
    assert torch.allclose(loaded.weight.data, torch.ones(1, 5, 10)), "Weights not restored"
    assert torch.allclose(loaded.bias.data, torch.ones(1, 5)), "Bias not restored"
    
    print("✓ Strict load init test passed")


def test_non_strict_init_load():
    """Test non-strict loading matches by position, not name."""
    print("\nTesting non-strict load by position...")
    
    # Create module with params
    module1 = SimpleNeatModule(10, 5)
    neat.Population(50, module1, Config(), verbose=False)
    with torch.no_grad():
        module1.weight[:] = torch.full_like(module1.weight.data, 2.0)
    
    serialized = module1.neat_dict()
    
    # Create module with same structure and load non-strictly
    module2 = SimpleNeatModule(10, 5)
    loaded = module2.load_neat_dict(
        serialized, 
        strict=False,
    )
    
    # Verify weights loaded by position (NeatParameter adds batch dimension)
    assert torch.allclose(loaded.weight.data, torch.full((1, 5, 10), 2.0)), "Non-strict load failed"
    
    print("✓ Non-strict load test passed")


def test_nested_module_serialization():
    """Test serialization of nested modules."""
    print("\nTesting nested module serialization...")
    
    module = NestedNeatModule(10, 8, NestedNeatModule(16, 16, NestedNeatModule(32, 32)))
    print(module)
    print(dict(module.named_parameters()))
    
    # Serialize
    serialized = module.neat_dict()
    pretty_print(serialized)
    # pretty_print(module.fc1.neat_dict())
    
    # Verify nested structure is captured
    arch = serialized['architecture']
    # print(arch['neat_modules'])
    assert 'neat_modules' in arch, "Missing nested modules info"
    
    # Should have multiple parameters in state_dict
    assert len(serialized['state_dict']) >= 4, "Should have at least 4 parameters (2 modules x 2 params each)"
    
    metadata = serialized['architecture']['neat_modules']
    state_dict = serialized['state_dict']
    population = serialized['population']
    pseudo_module = PseudoModule(metadata, state_dict, population)
    print(type(pseudo_module))
    # pretty_print(dict(pseudo_module.named_modules()))
    print(pseudo_module)
    # pretty_print(metadata)
    
    print("✓ Nested module serialization test passed")


def test_save_load_to_file():
    """Test save/load with actual files."""
    print("\nTesting save/load to files...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save module
        module1 = SimpleNeatModule(10, 5)
        neat.Population(50, module1, Config(), verbose=False)
        with torch.no_grad():
            module1.weight[:] = torch.full_like(module1.weight.data, 3.0)
        
        success, file_no = module1.save(
            filename="test_module",
            directory=tmpdir,
            file_no=0,
            replace=True,
            debug=True
        )
        assert success, "Save failed"
        # Note: file_no might be 0 or 1 depending on existing files
        
        # Load module
        module1.load(
            filename="test_module",
            directory=tmpdir,
            file_no=0,
            strict=True,
            debug=True
        )
        
        module2 = SimpleNeatModule(10, 5)
        module2.load_neat_dict(module1.neat_dict(), strict=True)
        
        # Verify (NeatParameter adds batch dimension)
        assert torch.allclose(module2.weight.data, torch.full((1, 5, 10), 3.0)), "Loaded weights don't match"
        
        print(f"✓ Save/load to file test passed (saved to {tmpdir})")


def test_strict_shape_mismatch():
    """Test that strict mode catches shape mismatches."""
    print("\nTesting strict mode shape mismatch detection...")
    
    module1 = SimpleNeatModule(10, 5)
    serialized = module1.neat_dict()
    
    # Try to load into wrong shape module
    module2 = SimpleNeatModule(10, 8)  # Different hidden size
    
    try:
        module2.load_neat_dict(
            serialized,
            strict=True,
        )
        assert False, "Should have raised ValueError for shape mismatch"
    except RuntimeError as e:
        assert any([v in str(e).lower() for v in ['match', 'mis']]), f"Wrong error message: {e}"
        print("✓ Shape mismatch detection works")


def test_population_preservation():
    """Test that popu;ation metadata is preserved through save/load."""
    print("\nTesting population metadata preservation...")
    
    module1 = SimpleNeatModule(10, 5)
    module1.genome_num = 67
    module1.mapping = {1: 0, 2: 1, 3: 2}
    # NOTE: The rest of the metadata is checked in the verification methods
    
    serialized = module1.neat_dict()
    
    module2 = SimpleNeatModule(10, 5)
    loaded = module2.load_neat_dict(
        serialized,
        strict=True
    )
    
    assert loaded.genome_num == 67, "genome_num not preserved"
    assert loaded.mapping == {1: 0, 2: 1, 3: 2}, "mapping not preserved"
    
    print("✓ Metadata preservation test passed")


if __name__ == "__main__":
    print("Running NeatModule save/load tests...\n")
    
    # test_state_dict_with_metadata()
    # test_strict_load()
    # test_strict_init_load()
    # test_non_strict_init_load()
    test_nested_module_serialization()
    # test_save_load_to_file()
    # test_strict_shape_mismatch()
    # test_population_preservation()
    
    print("\n✓ All NeatModule tests passed!")
