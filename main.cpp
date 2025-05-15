#include <iostream> 
#include <fstream>
#include <filesystem>
#include <string> 
#include <cstdlib>
#include <stdlib.h>
#include <stdexcept>
#include <stdio.h>
#include <armadillo>
#include <vector>

#include <nlohmann/json.hpp> // This is the JSON handling library
#include <armadillo> 

#include "basis.hpp"
#include "AO.hpp"
#include "CNDO.hpp"
#include "MDIntegrators.hpp"

// convenience definitions so the code is more readable
namespace fs = std::filesystem;
using json = nlohmann::json; 

void geometry_optimization(Molecule_basis& mol, CNDO& ourSCF, 
                          double step_size = 0.01, double tol = 1e-4, 
                          int max_steps = 100) {
    arma::mat gradient;
    double energy;
    int num_atoms = mol.getnum_atoms();
    
    for(int step=0; step<max_steps; step++) {
        // Compute energy and forces
        ourSCF.init();
        ourSCF.run();
        energy = ourSCF.getEnergy();
        gradient = ourSCF.getGradient();
        
        // Check convergence
        double grad_norm = arma::norm(gradient, "fro");
        std::cout << "Step " << step << ": Energy = " << energy 
                  << " eV, |Grad| = " << grad_norm << "\n";
        
        if(grad_norm < tol) {
            std::cout << "Converged after " << step << " steps\n";

            // Print XYZ format to terminal
            int num_atoms = mol.getnum_atoms();
            std::cout << num_atoms << "\n";
            std::cout << "Optimized geometry (Angstroms)\n";
            for(const auto& atom : mol.mAtoms) {
                arma::vec pos = atom.mAOs[0].get_R0();
                std::cout << atom.name << " "
                        << std::fixed << std::setprecision(6)
                        << pos(0) << " " << pos(1) << " " << pos(2) << "\n";
            }

            return;
        }
        
        // Update atomic positions using steepest descent
        for(int i=0; i<num_atoms; i++) {
            arma::vec current_pos = mol.mAtoms[i].mAOs[0].get_R0();
            arma::vec force = -gradient.col(i); // Negative gradient = force direction
            arma::vec new_pos = current_pos + step_size * force;
            mol.mAtoms[i].set_R0(new_pos);
        }
    }
    std::cout << "Reached maximum steps without convergence\n";

}



int main(int argc, char** argv){
    // check that a config file is supplied
    if (argc != 2){
        std::cerr << "Usage: " << argv[0] << " path/to/config.json" << std::endl; 
        return EXIT_FAILURE; 
    }
    
    // parse the config file 
    fs::path config_file_path(argv[1]);
    if (!fs::exists(config_file_path)){
        std::cerr << "Path: " << config_file_path << " does not exist" << std::endl; 
        return EXIT_FAILURE;
    }
    std::ifstream config_file(config_file_path); 
    json config = json::parse(config_file); 
    

    // extract the important info from the config file
    fs::path atoms_file_path = config["atoms_file_path"];
    fs::path output_file_path = config["output_file_path"];  
    int num_alpha_electrons = config["num_alpha_electrons"];
    int num_beta_electrons = config["num_beta_electrons"];
    
    // import Atoms from atoms_file_path
    arma::mat H_STO3G = basis_map["H"];
    arma::mat C_STO3G = basis_map["C"];

    Molecule_basis mol(std::string(atoms_file_path), H_STO3G, C_STO3G);
    
    int num_atoms = mol.getnum_atoms(); // You will have to replace this with the number of atoms in the molecule 
    int num_basis_functions = mol.getnumAOs(); // you will have to replace this with the number of basis sets in the molecule
    int num_3D_dims = 3; 

    // Your answers go in these objects 

    arma::mat Suv_RA(num_3D_dims, num_basis_functions * num_basis_functions); 
    // Ideally, this would be (3, n_funcs, n_funcs) rank-3 tensor
    // but we're flattening (n-funcs, n-atoms) into a single dimension (n-funcs ^ 2)
    // this is because tensors are not supported in Eigen and I want students to be able to 
    // submit their work in a consistent format
    arma::mat gammaAB_RA(num_3D_dims, num_atoms * num_atoms); 
    // This is the same story, ideally, this would be (3, num_atoms, num_atoms) instead of (3, num_atoms ^ 2)
    arma::mat gradient_nuclear(num_3D_dims, num_atoms);
    arma::mat gradient_electronic(num_3D_dims, num_atoms); 
    arma::mat gradient(num_3D_dims, num_atoms); 

    // most of the code goes here 


    CNDO ourSCF(mol, num_alpha_electrons, num_beta_electrons, 50, 1e-5);
    ourSCF.init();
    ourSCF.run();
    double Energy = ourSCF.getEnergy();
    
    
    gradient = ourSCF.getGradient(); // This call does work, so it needs to go first in this group of statements
    Suv_RA = ourSCF.Suv_RA;
    gammaAB_RA = ourSCF.gammaAB_RA; 
    gradient_nuclear = ourSCF.gradient_nuclear; 
    gradient_electronic = gradient - gradient_nuclear; 


    // You do not need to modify the code below this point 
    
    // Set print configs
    std::cout << std::fixed << std::setprecision(4) << std::setw(8) << std::right ; 

    // inspect your answer via printing
    Suv_RA.print("Suv_RA");
    gammaAB_RA.print("gammaAB_RA");
    gradient_nuclear.print("gradient_nuclear");
    gradient_electronic.print("gradient_electronic");
    gradient.print("gradient"); 



    /// print out numerical gradient 

    arma::mat numerical_gradient = ourSCF.getNumericalGradient(1e-6);
    numerical_gradient.print("Numerical Gradient");

    // perform geometry optimization 
    geometry_optimization(mol, ourSCF, 0.01, 1e-4, 1000);

    
    // After molecule creation:
    MDAtoms system(mol.getnum_atoms());
    // set the box as well 
    arma::vec3 box = {20.0, 20.0, 20.0};
    system.box_size = box;
    mol.setBox(box(0), box(1), box(2));

    
    // Initialize positions from molecule
    for(int i=0; i<mol.getnum_atoms(); i++) {
        system.positions.col(i) = mol.mAtoms[i].mAOs[0].get_R0();
    }
    
    // Set masses based on atom types
    system.masses.set_size(mol.getnum_atoms());
    for(int i=0; i<mol.getnum_atoms(); i++) {
        std::string elem = mol.mAtoms[i].name;
        if(elem == "H") system.masses[i] = 1.008;
        else if(elem == "C") system.masses[i] = 12.011;
        else if(elem == "N") system.masses[i] = 14.007;
        else if(elem == "O") system.masses[i] = 16.00;
        else if(elem == "F") system.masses[i] = 19.00;
        else throw std::runtime_error("Unsupported atom type");
    }
    
    // Initialize velocities and thermostat
    system.initializeVelocities(300.0); // 300K initial temp
    BerendsenThermostat thermostat(system, 300.0, 500.0); // 300K target, 100fs relaxation
    
    // MD parameters
    double dt = 0.1; // fs
    int n_steps = 1000;
    int traj_interval = 10;

    /*
    CNDO testSCF(mol, num_alpha_electrons, num_beta_electrons, 50, 1e-5);
    testSCF.init();
    testSCF.run();
    system.forces = testSCF.getForces();
    system.forces.raw_print("\nForces matrix (eV/Å):");
    */

    // Before MD loop:
    // CNDO mdSCF(mol, num_alpha_electrons, num_beta_electrons, 50, 1e-5);
    // mdSCF.init();
    // mdSCF.run();   // First proper SCF

    // Before MD loop:
    std::ofstream traj_file("trajectory.xyz");
    if (!traj_file) {
        std::cerr << "Error opening trajectory file!" << std::endl;
        return EXIT_FAILURE;
    }


    // Store initial densities
    system.density_alpha = ourSCF.getPa();  // Reuse optimized densities
    system.density_beta = ourSCF.getPb();
    system.forces = ourSCF.getForces();

    // ===== MD MAIN LOOP =====

    // Store initial state FROM GEOMETRY OPTIMIZATION
    system.density_alpha = ourSCF.getPa();  // Reuse optimized densities
    system.density_beta = ourSCF.getPb();
    system.forces = ourSCF.getForces();

    for(int step=0; step<n_steps; step++) {
        // --- 1. Update positions ---
        Verlet::step1(mol, system, dt);

        // --- 2. Electronic structure ---
        ourSCF.init();  // Critical: rebuild integrals for new positions
        ourSCF.setDensityMatrices(system.density_alpha, system.density_beta);
        ourSCF.run();

        // --- 3. Update forces/densities ---
        system.forces = -ourSCF.getGradient();
        system.density_alpha = ourSCF.getPa();
        system.density_beta = ourSCF.getPb();

        // --- 4. Complete velocity update ---
        Verlet::step2(mol, system, dt);

        // --- 5. Thermostat ---
        thermostat.apply(dt);

        // --- 6. Trajectory output ---
        if(step % traj_interval == 0) {
            traj_file << mol.getnum_atoms() << "\n";
            traj_file << "Lattice=\""
                      << mol.getLatticeVector(0)(0) << " " << mol.getLatticeVector(0)(1) << " " << mol.getLatticeVector(0)(2) << " "
                      << mol.getLatticeVector(1)(0) << " " << mol.getLatticeVector(1)(1) << " " << mol.getLatticeVector(1)(2) << " "
                      << mol.getLatticeVector(2)(0) << " " << mol.getLatticeVector(2)(1) << " " << mol.getLatticeVector(2)(2) << "\" "
                      << "Step=" << step << " Energy=" << ourSCF.getEnergy()  // Now using ourSCF
                      << " Temperature=" << thermostat.current_temperature() << "\n";
            
            for(int i=0; i<mol.getnum_atoms(); i++) {
                arma::vec pos = mol.mAtoms[i].mAOs[0].get_R0();
                traj_file << mol.mAtoms[i].name << " "
                        << pos(0) << " " << pos(1) << " " << pos(2) << "\n";
            }
        }
    }


      
}  
