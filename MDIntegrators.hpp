#pragma once
#include <armadillo>
#include <cmath>

// Physical constants and unit conversions
const double EV_TO_HARTREE = 0.0367493;
const double ANGSTROM_TO_BOHR = 1.88973;
const double FS_TO_ATOMIC_TIME = 41.341372; // 1 fs = 41.341 atomic time units
const double DALTON_TO_ELECTRON_MASS = 1822.888486; // 1 Da = 1822.888486 me
const double BOHR_TO_ANGSTROM = 0.529177; // or 1.0 / ANGSTROM_TO_BOHR



// Random number generator (Marsaglia's MWC)
unsigned int m_u = 521288629, m_v = 362436069;
double brng() {
    m_v = 36969*(m_v & 65535) + (m_v >> 16);
    m_u = 18000*(m_u & 65535) + (m_u >> 16);
    return ((m_v << 16) + m_u) * 2.328306435996595e-10;
}

// Gaussian distribution generator (Box-Muller)
double gaussdist() {
    static bool has_saved = false;
    static double saved;
    
    if (has_saved) {
        has_saved = false;
        return saved;
    }
    
    double x, y, r;
    do {
        x = 2.0 * brng() - 1.0;
        y = 2.0 * brng() - 1.0;
        r = x*x + y*y;
    } while (r >= 1.0 || r == 0.0);
    
    double d = sqrt(-2.0 * log(r)/r);
    saved = x * d;
    has_saved = true;
    return y * d;
}

class MDAtoms {
public:
    arma::mat positions;    // 3xN (Å)
    arma::mat velocities;   // 3xN (Å/fs)
    arma::mat forces;       // 3xN (eV/Å)
    arma::vec masses;       // N (Da)
    arma::mat density_alpha; // Persistent density matrices
    arma::mat density_beta;  // Between MD steps
    arma::vec3 box_size;    // Box dimensions (Å)
    bool pbc[3] = {true, true, true};  // Enable PBC in x/y/z directions

    MDAtoms(int num_atoms) : 
        positions(3, num_atoms, arma::fill::zeros),
        velocities(3, num_atoms, arma::fill::zeros),
        forces(3, num_atoms, arma::fill::zeros),
        masses(num_atoms, arma::fill::ones),
        density_alpha(arma::size(forces)),
        density_beta(arma::size(forces)),
        box_size{20.0, 20.0, 20.0} {}  // Default 20Å cubic box

    void initializeVelocities(double target_temp) {
        // Generate Gaussian-distributed velocities
        velocities.imbue([&]() { return gaussdist(); });
        
        // Remove center-of-mass velocity
        arma::vec vCM = arma::sum(velocities.each_row() % masses.t(), 1) / arma::accu(masses);
        velocities.each_col() -= vCM;
        
        // Scale to target temperature (KE = 3/2 NkBT)
        double kinetic = 0.5 * arma::accu(masses % arma::sum(velocities % velocities, 0).t());
        double current_temp = kinetic * 4.888821e-4; // Da*(Å/fs)^2 → K
        velocities *= sqrt(target_temp / current_temp);
    }
};


class BerendsenThermostat {
    MDAtoms& atoms;
    double target_temp;
    double tau; // Relaxation time (fs)
    
    // Helper to calculate instantaneous temperature
    double computeTemperature() const {
        double kinetic = 0.5 * arma::accu(atoms.masses % 
            arma::sum(atoms.velocities % atoms.velocities, 0).t());
        return kinetic * 4.888821e-4; // Converts to Kelvin
    }
    
public:
    BerendsenThermostat(MDAtoms& system, double temp, double relaxation_time) : 
        atoms(system), target_temp(temp), tau(relaxation_time) {}
    
    void apply(double dt) {
        double current_temp = computeTemperature();
        double lambda = sqrt(1 + (dt/tau)*(target_temp/current_temp - 1));
        atoms.velocities *= lambda;
    }
    
    // New function to access current temperature
    double current_temperature() const {
        return computeTemperature();
    }
};



// New constraint function mimicking geometry optimizer behavior
void applyGeometryConstraints(Molecule_basis& mol, MDAtoms& atoms) {
    const double force_tol = 1e-4; // Match optimization tolerance
    const double max_step = 0.01; // Similar to optimization step size
    
    // Calculate current bond deviations
    arma::vec bond_errors;
    for(const auto& bond : mol.bonds) {
        arma::vec3 dr = mol.minimum_image(
            mol.mAtoms[bond.atom1].mAOs[0].get_R0() - 
            mol.mAtoms[bond.atom2].mAOs[0].get_R0()
        );
        double current_len = arma::norm(dr);
        bond_errors = arma::join_vert(bond_errors, arma::vec({current_len - bond.eq_length}));
    }

    // Apply iterative correction (similar to optimization loop)
    while(arma::max(arma::abs(bond_errors)) > force_tol) {
        // Calculate constraint forces
        arma::mat constraint_forces(3, mol.mAtoms.size(), arma::fill::zeros);
        
        for(size_t i=0; i<mol.bonds.size(); ++i) {
            const auto& bond = mol.bonds[i];
            arma::vec3 dr = mol.minimum_image(
                mol.mAtoms[bond.atom1].mAOs[0].get_R0() - 
                mol.mAtoms[bond.atom2].mAOs[0].get_R0()
            );
            double current_len = arma::norm(dr);
            arma::vec3 force_dir = dr / (current_len + 1e-8);
            
            // Apply harmonic constraint (similar to optimization gradient)
            double k = 0.1; // Stiffness parameter
            double force_mag = k * (current_len - bond.eq_length);
            
            constraint_forces.col(bond.atom1) -= force_mag * force_dir;
            constraint_forces.col(bond.atom2) += force_mag * force_dir;
        }
        
        // Apply constrained position updates
        for(size_t i=0; i<mol.mAtoms.size(); ++i) {
            arma::vec R = mol.mAtoms[i].mAOs[0].get_R0();
            R += max_step * constraint_forces.col(i) / atoms.masses(i);
            mol.mAtoms[i].set_R0(R);
        }
        
        // Recalculate errors
        bond_errors.zeros();
        for(const auto& bond : mol.bonds) {
            arma::vec3 dr = mol.minimum_image(
                mol.mAtoms[bond.atom1].mAOs[0].get_R0() - 
                mol.mAtoms[bond.atom2].mAOs[0].get_R0()
            );
            double current_len = arma::norm(dr);
            bond_errors = arma::join_vert(bond_errors, arma::vec({current_len - bond.eq_length}));
        }
    }
}




namespace Verlet {
void step1(Molecule_basis& mol, MDAtoms& atoms, double dt) {
    // Store original molecular center
    mol.detectBonds(2.0); // Add this line

    arma::vec3 mol_center = arma::mean(atoms.positions, 1);
    
    // Update velocities and unwrapped positions
    for(size_t i = 0; i < mol.mAtoms.size(); ++i) {
        arma::vec3 a = atoms.forces.col(i) / atoms.masses(i);
        atoms.velocities.col(i) += 0.5 * a * dt;
        
        // Update position using minimum image convention
        arma::vec R = mol.mAtoms[i].mAOs[0].get_R0();
        R += atoms.velocities.col(i) * dt;
        mol.mAtoms[i].set_R0(R);
    }

    // Maintain molecular geometry using optimization-style constraints
    applyGeometryConstraints(mol, atoms);

    // Calculate new molecular center after updates
    arma::vec3 new_center = arma::mean(atoms.positions, 1);
    
    // Apply PBC to entire molecule as a whole
    arma::vec3 box = mol.box_size;
    arma::vec3 shift = (new_center - mol_center) - mol.minimum_image(new_center - mol_center);
    
    for(auto& atom : mol.mAtoms) {
        arma::vec R = atom.mAOs[0].get_R0();
        R -= shift;
        atom.set_R0(R);
    }

    // Final PBC application with molecular continuity
    mol.applyPBC();
    
    // Sync positions with MDAtoms
    for(size_t i = 0; i < mol.mAtoms.size(); ++i) {
        atoms.positions.col(i) = mol.mAtoms[i].mAOs[0].get_R0();
    }
}


void step2(Molecule_basis& mol, MDAtoms& atoms, double dt) {
    for(size_t i = 0; i < mol.mAtoms.size(); ++i) {
        arma::vec3 a_new = atoms.forces.col(i) / atoms.masses(i);
        atoms.velocities.col(i) += 0.5 * a_new * dt;
    }
}
}
