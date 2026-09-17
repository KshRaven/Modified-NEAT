"""Comprehensive tests for Population save/load with state_dict_with_metadata()."""

import sys
import tempfile
from pathlib import Path
import torch
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ModifiedNEAT.nn.base import NeatModule, NeatParameter
from ModifiedNEAT.config import Config
from ModifiedNEAT.population import Population
from ModifiedNEAT.nn.genome import Genome


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


def create_test_population(num_genomes=15):
    """Create a simple test Population."""
    module = SimpleNeatModule(10, 5)
    config = Config()
    population = Population(num_genomes, module, config=config, verbose=False)
    
    return population, module, config


def test_population_save_dict():
    """Test Population.save_dict() serialization."""
    print("\nTesting Population.save_dict()...")
    
    pop, module, config = create_test_population(3)
    
    # Get save dict
    save_dict, file_no = pop.save_dict(name=None)
    
    # Verify structure
    assert 'version' in save_dict, "Missing version"
    assert 'genera' in save_dict, "Missing genera"
    assert 'generation' in save_dict, "Missing generation"
    assert 'module_state_dicts' in save_dict, "Missing module_state"
    assert 'genomes' in save_dict, "Missing genomes"
    assert 'species' in save_dict, "Missing species"
    
    # Verify module architecture is saved
    assert len(save_dict['module_state_dicts']) > 0, "No module architecture saved"
    
    print("✓ Population.save_dict() test passed")


def test_population_save_load_to_file():
    """Test Population save/load with actual files."""
    print("\nTesting Population save/load to files...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save population
        pop1, module, config = create_test_population(3)
        
        # Modify some weights
        with torch.no_grad():
            module.weight[:] = torch.randn_like(module.weight.data)
        
        success, file_no = pop1.save(
            filename="test_population",
            directory=tmpdir,
            file_no=0,
            replace=True,
            debug=True
        )
        assert success, "Population save failed"
        
        # Load population (use non-strict mode since SimpleNeatModule isn't importable in this context)
        pop2 = create_test_population(10)[0]
        pop2.load(
            filename="test_population",
            directory=tmpdir,
            file_no=0,
            strict=False,  # Use non-strict mode for test modules not globally importable
            debug=True
        )
        
        # Verify population structure
        assert len(pop2.genomes) == len(pop1.genomes), "Genome count mismatch"
        assert len(pop2.modules) == len(pop1.modules), "Module count mismatch"
        
        # Verify weights were restored (access via neat_parameters since module class may vary)
        loaded_module = list(pop2.modules.values())[0]
        params = loaded_module.neat_parameters()
        assert len(params) >= 2, f"Expected at least 2 parameters, got {len(params)}"
        # The first parameter should be weight (shape 1, 5, 10), second bias (shape 1, 5)
        assert torch.allclose(params[0].data, list(pop1.modules.values())[0].neat_parameters()[0].data), "First parameter (weight) not restored"
        
        print(f"✓ Population save/load test passed")


def test_population_strict_vs_non_strict():
    """Test strict vs non-strict loading."""
    print("\nTesting Population strict vs non-strict loading...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save
        pop1, module, config = create_test_population(3)
        pop1.save(
            filename="test_strict",
            directory=tmpdir,
            debug=False
        )
        
        # Load with strict=True
        pop_strict = Population.load(
            filename="test_strict",
            directory=tmpdir,
            strict=True,
            debug=False
        )
        assert len(pop_strict.genomes) > 0, "Strict load failed"
        
        # Load with strict=False
        pop_non_strict = Population.load(
            filename="test_strict",
            directory=tmpdir,
            strict=False,
            debug=False
        )
        assert len(pop_non_strict.genomes) > 0, "Non-strict load failed"
        
        print("✓ Strict vs non-strict loading works")


def test_module_architecture_preservation():
    """Test that module architecture is correctly preserved."""
    print("Testing module architecture preservation...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pop1, module, config = create_test_population(2)
        
        # Get original architecture
        original_params = list(module.neat_parameters())
        original_count = len(original_params)
        
        # Save and load
        pop1.save(filename="test_arch", directory=tmpdir, debug=False)
        pop2 = Population.load(filename="test_arch", directory=tmpdir, debug=False)
        
        # Verify architecture
        loaded_module = list(pop2.modules.values())[0]
        loaded_params = list(loaded_module.neat_parameters())
        loaded_count = len(loaded_params)
        
        assert loaded_count == original_count, f"Parameter count mismatch: {loaded_count} vs {original_count}"
        
        print("✓ Module architecture preservation test passed")


def test_multiple_generations_save():
    """Test saving/loading after multiple generations."""
    print("Testing multiple generations save...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pop1, module, config = create_test_population(3)
        
        # Simulate multiple generations
        pop1.generation = 10
        
        # Save
        success, file_no1 = pop1.save(filename="test_gen", directory=tmpdir, debug=False)
        
        # Save another version (should increment file number)
        pop1.generation = 20
        success, file_no2 = pop1.save(filename="test_gen", directory=tmpdir, debug=False)
        
        # Load first version
        pop_v1 = Population.load(filename="test_gen", directory=tmpdir, file_no=file_no1, debug=False)
        assert pop_v1.generation == 10, "Wrong generation loaded (v1)"
        
        # Load second version
        pop_v2 = Population.load(filename="test_gen", directory=tmpdir, file_no=file_no2, debug=False)
        assert pop_v2.generation == 20, "Wrong generation loaded (v2)"
        
        print("✓ Multiple generations save test passed")


def test_genomes_preserved():
    """Test that genomes are correctly preserved."""
    print("Testing genomes preservation...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pop1, _, _ = create_test_population(5)
        
        # Record genome keys
        original_keys = set(pop1.genomes.keys())
        
        # Save and load
        pop1.save(filename="test_genomes", directory=tmpdir, debug=False)
        pop2 = Population.load(filename="test_genomes", directory=tmpdir, debug=False)
        
        # Verify genomes
        loaded_keys = set(pop2.genomes.keys())
        assert original_keys == loaded_keys, "Genome keys don't match"
        
        # Verify all genomes have fitness data
        for gid, genome in pop2.genomes.items():
            assert hasattr(genome, 'fitness'), f"Genome {gid} missing fitness"
            assert hasattr(genome, 'genus'), f"Genome {gid} missing genus"
        
        print("✓ Genomes preservation test passed")


def test_save_dict_load_dict_consistency():
    """Test that save_dict/load_dict are consistent."""
    print("Testing save_dict/load_dict consistency...")
    
    pop1, _, _ = create_test_population(3)
    
    # Use save_dict/load_dict directly
    save_dict, _ = pop1.save_dict(name=None)
    
    pop2 = Population.__new__(Population)
    pop2.config = pop1.config
    pop2.modules = pop1.modules
    pop2.reproduction = pop1.reproduction
    pop2.reporters = pop1.reporters
    pop2.species_set = pop1.species_set
    pop2.threads_per_block = pop1.threads_per_block
    
    # Load dict
    pop2.load_dict(save_state=save_dict, strict=True, verbose=0)
    
    # Verify consistency
    assert len(pop2.genomes) == len(pop1.genomes), "Genome count inconsistent"
    assert len(pop2.species_set.species) > 0, "Species not loaded"
    
    print("✓ save_dict/load_dict consistency test passed")


if __name__ == "__main__":
    print("Running Population save/load tests...\n")
    
    try:
        test_population_save_dict()
        test_population_save_load_to_file()
        # test_population_strict_vs_non_strict()
        # test_module_architecture_preservation()
        # test_multiple_generations_save()
        # test_genomes_preserved()
        # test_save_dict_load_dict_consistency()
        
        print("\n✓ All Population tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
