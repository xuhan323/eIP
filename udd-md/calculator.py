from ase.calculators.calculator import Calculator, all_changes
import numpy as np




class MLCalculator_schnet(Calculator):
    implemented_properties = ["energy", "forces", 
                              "forces_biased_mean", "forces_biased_std", "forces_biased_max",
                              "uncertainty_mean", "uncertainty_std"]

    def __init__(
        self,
        model,
        energy_scale=1,
        forces_scale=1,
        # energy_scale=1.0,
        # forces_scale=1.0,
        #debug,orginal=1.0
#        stress_scale=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.model = model
        self.model_device = next(model.parameters()).device
        self.cutoff = model.cutoff

        self.energy_scale = energy_scale
        self.forces_scale = forces_scale
#        self.stress_scale = stress_scale

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """
        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): do not use this, no functionality
            system_changes (list of str): List of changes for ASE.
        """
        # First call original calculator to set atoms attribute
        # (see https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/calculator.html#Calculator)
        if atoms is not None:
            self.atoms = atoms.copy()       
        from torch_geometric.data import Data, DataLoader
        import torch
        data = Data(
            pos=torch.as_tensor(self.atoms.positions,dtype=torch.float32),
            z=torch.as_tensor(atoms.numbers),
            cell=torch.as_tensor(atoms.cell[:]).unsqueeze(0),
            # energy=energy,
            # force=torch.as_tensor(forces, dtype=torch.float64),
            natoms=len(atoms.numbers),
        )
        data_list =[]
        # print(data.z)
        data_list.append(data)
        atomic_data = DataLoader(data_list,4,shuffle=False)
        
        batch_data=list(atomic_data)[0]
        # device = 'cuda:0'
        batch_data = batch_data.to(self.model_device)
        model_results = self.model(batch_data)

        results = {}
        # print(model_results)
        # print(model_results["forces"][0])
        # Convert outputs to calculator format
        results["forces"] = (
            model_results["forces"].detach().cpu().numpy() * self.forces_scale
        )
        results["energy"] = (
            model_results["energy"][0].detach().cpu().numpy().item()
            * self.energy_scale
        )
        results["forces_biased_mean"] = (
            torch.mean(abs(model_results["forces_biased"])).detach().cpu().numpy() * self.forces_scale
        )
        results["forces_biased_max"] = (
            torch.max(abs(model_results["forces_biased"])).detach().cpu().numpy() * self.forces_scale
        )
        results["forces_biased_std"] = (
            torch.std(abs(model_results["forces_biased"])).detach().cpu().numpy() * self.forces_scale
        )
        results["uncertainty_mean"] = (
            torch.mean(model_results["uncertainty"]).detach().cpu().numpy() * self.forces_scale
        )
        results["uncertainty_std"] = (
            torch.std(model_results["uncertainty"]).detach().cpu().numpy() * self.forces_scale
        )
#            results["stress"] = (
#                model_results["stress"][0].detach().cpu().numpy() * self.stress_scale
#            )
#         atoms.info["ll_out"] = {
#             k: v.detach().cpu().numpy() for k, v in model_results["ll_out"].items()
#         }
        if model_results.get("fps"):
            atoms.info["fps"] = model_results["fps"].detach().cpu().numpy()
    
        self.results = results

class EnsembleCalculator_schnet(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        models,
        energy_scale=1,
        forces_scale=1,
        #debug,original=1.0
#        stress_scale=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.models = models
        self.model_device = next(models[0].parameters()).device
        self.cutoff = models[0].cutoff
        self.energy_scale = energy_scale
        self.forces_scale = forces_scale
#       self.stress_scale = stress_scale

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """
        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): do not use this, no functionality
            system_changes (list of str): List of changes for ASE.
        """
        # First call original calculator to set atoms attribute
        # (see https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/calculator.html#Calculator)
        if atoms is not None:
            self.atoms = atoms.copy()       
        from torch_geometric.data import Data, DataLoader
        import torch
        data = Data(
            pos=torch.as_tensor(self.atoms.positions, dtype=torch.float64),
            z=torch.as_tensor(atoms.numbers, dtype=torch.int),
            cell=torch.as_tensor(atoms.cell[:]),
            # energy=energy,
            # force=torch.as_tensor(forces, dtype=torch.float64),
            # natoms=len(atoms.numbers),
        )
        atomic_data = DataLoader(data,1,shuffle=False)
        # model_inputs = self.ase_data_reader(self.atoms)
        # model_inputs = {
        #     k: v.to(self.model_device) for (k, v) in model_inputs.items()
        # }
        batch_data=list(atomic_data)[0]
        
        predictions = {'energy': [], 'forces': []}
        for model in self.models:
            model_results = self.model(batch_data)
            predictions['energy'].append(model_results["energy"][0].detach().cpu().numpy().item() * self.energy_scale)
            predictions['forces'].append(model_results["forces"].detach().cpu().numpy() * self.forces_scale)

        results = {"energy": np.mean(predictions['energy'])}
        results["forces"] = np.mean(np.stack(predictions['forces']), axis=0)

        ensemble = {
            'energy_var': np.var(predictions['energy']),
            'forces_var': np.var(np.stack(predictions['forces']), axis=0),
            'forces_l2_var': np.var(np.linalg.norm(predictions['forces'], axis=2), axis=0),
        }

        results['ensemble'] = ensemble

        self.results = results
    
class MLCalculator_tip3p(Calculator):
    implemented_properties = ["energy", "forces"]
    nolable = True
    pcpot = None

    def __init__(
        self,
        model,
        energy_scale=1,
        forces_scale=1,
        position_scale=1,
        rc: float = 5.0,
        width: float = 1,
        # energy_scale=1.0,
        # forces_scale=1.0,
        #debug,orginal=1.0
#        stress_scale=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.model = model
        self.model_device = next(model.parameters()).device
        self.cutoff = model.cutoff

        self.energy_scale = energy_scale
        self.forces_scale = forces_scale
        self.position_scale = position_scale
        self.rc = rc
        self.width = width
#        self.stress_scale = stress_scale

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """
        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): do not use this, no functionality
            system_changes (list of str): List of changes for ASE.
        """
        # First call original calculator to set atoms attribute
        # (see https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/calculator.html#Calculator)
        if atoms is not None:
            self.atoms = atoms.copy()
        from ase.calculators.tip3p import TIP3P       
        from torch_geometric.data import Data, DataLoader
        import torch
        data = Data(
            pos=torch.as_tensor(self.atoms.positions*self.position_scale,dtype=torch.float32),
            z=torch.as_tensor(atoms.numbers),
            cell=torch.as_tensor(atoms.cell[:]*self.position_scale).unsqueeze(0),
            # z=torch.as_tensor(atoms.numbers),
            # cell=torch.as_tensor(atoms.cell[:]).unsqueeze(0),
            # energy=energy,
            # force=torch.as_tensor(forces, dtype=torch.float64),
            natoms=len(atoms.numbers),
        )
        data_list =[]
        # print(data.z)
        data_list.append(data)
        atomic_data = DataLoader(data_list,2,shuffle=False)

        batch_data=list(atomic_data)[0]

        
        # device = 'cuda:0'
        batch_data = batch_data.to(self.model_device)
        model_results = self.model(batch_data)

        # debug uncertainty remove later
        # uncertainty = run.val(self, model=self.model, data_loader=atomic_data, energy_and_force=True, p=100000, evaluation=True, device=self.model_device)
        # print(uncertainty)
        # ###############

        tip3p_results = TIP3P.calculate(self, atoms=atoms, properties=['energy', 'forces'])
        print(model_results["forces_biased"])
        results = {}
        # print(tip3p_results)
        # print(model_results)
        print(model_results["forces"])
        # Convert outputs to calculator format
        results["forces"] = (
            tip3p_results["forces"] + model_results["forces_biased"].detach().cpu().numpy() * self.forces_scale
        )
        results["energy"] = (
            tip3p_results["energy"] + model_results["energy_biased"][0].detach().cpu().numpy().item()
            * self.energy_scale
        )
        import numpy
        # print(numpy.mean(abs(tip3p_results["forces"])), numpy.var(abs(tip3p_results["forces"])))
        # print(numpy.mean(abs(model_results["forces_biased"].detach().cpu().numpy())), numpy.var(model_results["forces_biased"].detach().cpu().numpy()))
        exit()
#            results["stress"] = (
#                model_results["stress"][0].detach().cpu().numpy() * self.stress_scale
#            )
#         atoms.info["ll_out"] = {
#             k: v.detach().cpu().numpy() for k, v in model_results["ll_out"].items()
#         }
        if model_results.get("fps"):
            atoms.info["fps"] = model_results["fps"].detach().cpu().numpy()
    
        self.results = results

class EnsembleCalculator_tip3p(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        models,
        energy_scale=1,
        forces_scale=1,
        position_scale=0.1,
        #debug,original=1.0
#        stress_scale=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.models = models
        self.model_device = next(models[0].parameters()).device
        self.cutoff = models[0].cutoff
        self.energy_scale = energy_scale
        self.forces_scale = forces_scale
        self.position_scale = position_scale
#       self.stress_scale = stress_scale

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """
        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): do not use this, no functionality
            system_changes (list of str): List of changes for ASE.
        """
        # First call original calculator to set atoms attribute
        # (see https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/calculator.html#Calculator)
        if atoms is not None:
            self.atoms = atoms.copy()       
        from torch_geometric.data import Data, DataLoader
        import torch
        data = Data(
            pos=torch.as_tensor(self.atoms.positions*self.position_scale, dtype=torch.float64),
            z=torch.as_tensor(self.atoms.numbers, dtype=torch.int),
            cell=torch.as_tensor(self.atoms.cell[:]*self.position_scale),
            # energy=energy,
            # force=torch.as_tensor(forces, dtype=torch.float64),
            # natoms=len(atoms.numbers),
        )
        atomic_data = DataLoader(data,1,shuffle=False)
        # model_inputs = self.ase_data_reader(self.atoms)
        # model_inputs = {
        #     k: v.to(self.model_device) for (k, v) in model_inputs.items()
        # }
        batch_data=list(atomic_data)[0]
        
        predictions = {'energy': [], 'forces': []}
        for model in self.models:
            model_results = self.model(batch_data)
            predictions['energy'].append(model_results["energy"][0].detach().cpu().numpy().item() * self.energy_scale)
            predictions['forces'].append(model_results["forces"].detach().cpu().numpy() * self.forces_scale)

        results = {"energy": np.mean(predictions['energy'])}
        results["forces"] = np.mean(np.stack(predictions['forces']), axis=0)

        ensemble = {
            'energy_var': np.var(predictions['energy']),
            'forces_var': np.var(np.stack(predictions['forces']), axis=0),
            'forces_l2_var': np.var(np.linalg.norm(predictions['forces'], axis=2), axis=0),
        }

        results['ensemble'] = ensemble

        self.results = results