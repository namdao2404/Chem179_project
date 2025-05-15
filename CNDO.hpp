#pragma once 

#include <stdio.h>
#include <math.h>
#include <cassert>
#include <unordered_map>
#include <iostream>
#include <vector>
#include <iomanip>

#include <armadillo>

#include "AO.hpp"

using namespace std;

class SCF{
  public:
      virtual int init()=0;
      virtual int run()=0;
      virtual arma::mat getPa()=0;
      virtual arma::mat getPb()=0;
      virtual double getEnergy()=0;
};

class CNDO: public SCF{
  private:
      Molecule_basis &mol;
      int dim, max_iter;
      int p, q; //num of alpha and beta electrons
      double tol;
      arma::mat S;
      arma::mat gamma;
      arma::mat H_core;
      arma::mat Pa, Pb;
      arma::mat Ga, Gb;
      arma::mat Ca, Cb;
      arma::vec Ea, Eb;
      double Ee, Ec, Etotal;
  public:
      CNDO(Molecule_basis &mol_i, int _p, int _q, int max_it, double tolerence);
      virtual int init();
      int updateG();
      virtual int run();
      virtual arma::mat getPa() {return Pa;}
      virtual arma::mat getPb() {return Pb;}
      virtual double getEnergy();
      arma::mat getGradient();
      arma::mat Suv_RA; 
      arma::mat gammaAB_RA; 
      arma::mat gradient_nuclear; 

      // add new lines to verify 
      arma::mat getNumericalGradient(double epsilon = 1e-6);  // Add this line
      arma::mat numerical_gradient;

      // the force is just the negative of the gradient 
      arma::mat getForces() { return -getGradient(); }  // Add this line

      // Add these setters for density matrices
      void setPa(const arma::mat& new_Pa) { Pa = new_Pa; }
      void setPb(const arma::mat& new_Pb) { Pb = new_Pb; }
      void setDensityMatrices(const arma::mat& new_Pa, const arma::mat& new_Pb) {
          Pa = new_Pa;
          Pb = new_Pb;
      }
};


struct CNDO_para
{
  std::map<std::string, double> IA;
  double beta;
} C_para = {{{"s", 14.051}, {"p", 5.572}}, -21.}, H_para{{{"s", 7.176}}, -9.};
CNDO_para N_para = {{{"s", 19.316}, {"p", 7.275}}, -25.};
CNDO_para O_para = {{{"s", 25.390}, {"p", 9.111}}, -31.};
CNDO_para F_para = {{{"s", 32.272}, {"p", 11.080}}, -39.};

std::map<std::string, CNDO_para> CNDO_para_map{
    {"H", H_para},
    {"C", C_para},
    {"N", N_para},
    {"O", O_para},
    {"F", F_para},
};

CNDO::CNDO(Molecule_basis &mol_i, int _p, int _q, int max_it, double tolerance) : mol(mol_i), max_iter(max_it), tol(tolerance), p(_p), q(_q)
{
  dim = mol.getnumAOs();
  Pa = arma::zeros(dim, dim);
  Pb = arma::zeros(dim, dim);
  Ga = arma::zeros(dim, dim);
  Gb = arma::zeros(dim, dim);
  Ca.set_size(dim, dim);
  Cb.set_size(dim, dim);
  Ea.set_size(dim);
  Eb.set_size(dim);

  H_core.set_size(dim, dim);
  S.set_size(dim, dim);
  mol.eval_OVmat(S);
  int num_atoms = mol.getnum_atoms();
  gamma.set_size(num_atoms, num_atoms);
  mol.eval_gammamat(gamma);
  // std::cout << std::setprecision(3);
  // gamma.print("gamma");
  // S.print("Overlap");

  // cout << " p =  " << p << " q =  " << q  << endl;
  Ee = 0.;
  Etotal = 0.;
  Ec = mol.eval_nuc_E();
}

int CNDO::init()
{

  // Only initialize density matrices if empty ===============
  if(Pa.is_empty()) Pa = arma::zeros(dim, dim);
  if(Pb.is_empty()) Pb = arma::zeros(dim, dim);
  // =========================================================



  int num_atoms = mol.getnum_atoms();
  //Do H_core part
  size_t k_AO = 0;
  for (size_t k = 0; k < num_atoms; k++)
  {
    Atom &A_Atom = mol.mAtoms[k];
    double ZA = double(A_Atom.VAN);
    double gammaAA = gamma(k, k);
    CNDO_para A_para = CNDO_para_map[A_Atom.name];
    for (auto ao_A : A_Atom.mAOs)
    {
      H_core(k_AO, k_AO) = -A_para.IA[ao_A.get_lable()] - (ZA - 0.5) * gammaAA;
      // H_core.print("H_core");
      size_t j_AO = 0;
      for (size_t j = 0; j < num_atoms; j++)
      {
        Atom &B_Atom = mol.mAtoms[j];
        CNDO_para B_para = CNDO_para_map[B_Atom.name];
        double averagebeta = (A_para.beta + B_para.beta) / 2.;
        if (k != j)
          H_core(k_AO, k_AO) -= double(B_Atom.VAN) * gamma(k, j);
        // cout<< k_AO << " " << H_core(k_AO, k_AO)<< endl;
        for (auto ao : B_Atom.mAOs)
        {
          if (k_AO != j_AO)
            H_core(k_AO, j_AO) = averagebeta * S(k_AO, j_AO);
          j_AO++;
        }
      }
      k_AO++;
    }
  }
  if (k_AO != dim)
  {
    cout << "warn! the number of AOs is wrong." << endl;
    return 1;
  }
  // H_core.print("H_core");

  return 0;
}

int CNDO::updateG()
{
  // std::cout << std::setprecision(3);
  int num_atoms = mol.getnum_atoms();
  arma::vec P_t = arma::zeros(num_atoms);
  size_t k_AO = 0;
  for (size_t k = 0; k < num_atoms; k++)
    for (auto ao : mol.mAtoms[k].mAOs)
    {
      P_t(k) += Pa(k_AO, k_AO) + Pb(k_AO, k_AO);
      k_AO++;
    }
  // P_t.print("P_t");
  k_AO = 0;
  for (size_t k = 0; k < num_atoms; k++)
  {
    double gammaAA = gamma(k, k);
    for (auto ao_A : mol.mAtoms[k].mAOs)
    {
      Ga(k_AO, k_AO) = (P_t(k) - Pa(k_AO, k_AO)) * gammaAA;
      Gb(k_AO, k_AO) = (P_t(k) - Pb(k_AO, k_AO)) * gammaAA;
      size_t j_AO = 0;
      for (size_t j = 0; j < num_atoms; j++)
      {
        double gammaAB = gamma(k, j);
        if (k != j)
        {
          Ga(k_AO, k_AO) += P_t(j) * gammaAB;
          Gb(k_AO, k_AO) += P_t(j) * gammaAB;
        }
        for (auto ao : mol.mAtoms[j].mAOs)
        {
          if (k_AO != j_AO)
          {
            Ga(k_AO, j_AO) = -gammaAB * Pa(k_AO, j_AO);
            Gb(k_AO, j_AO) = -gammaAB * Pb(k_AO, j_AO);
            // cout << k_AO << "  " << j_AO<< "  " << gammaAB<<"  " << Pa(k_AO, j_AO)<< "  " << Ga(k_AO, j_AO)<< endl;
          }
          j_AO++;
        }
      }
      k_AO++;
    }
  }
  // Ga.print("Ga");
  // Gb.print("Gb");
  return 0;
}

int CNDO::run()
{
  arma::mat Fa = H_core + Ga;
  arma::mat Fb = H_core + Gb;
  arma::mat Pa_old, Pb_old;
  size_t k = 0;
  for (; k < max_iter; k++)
  {
    // cout << "Iteration: " << k << endl;
    // Pa.print("Pa");
    // Ga.print("Ga");
    // Fa.print("Fa");
    // Fb.print("Fb");
    Pa_old = Pa;
    Pb_old = Pb;
    arma::eig_sym(Ea, Ca, Fa);
    arma::eig_sym(Eb, Cb, Fb);
    // cout << "after solving eigen equation: " << k << endl;
    // Ca.print("Ca");
    // Cb.print("Cb");
    // cout << " p =  " << p << " q =  " << q  << endl;
    Pa = Ca.cols(0, p - 1) * Ca.cols(0, p - 1).t();
    if (q > 0)
      Pb = Cb.cols(0, q - 1) * Cb.cols(0, q - 1).t();
    else
      Pb.zeros();
    if (arma::approx_equal(Pa, Pa_old, "absdiff", tol) && arma::approx_equal(Pb, Pb_old, "absdiff", tol))
      break;
    // Pa.print("Pa_new");
    // Pb.print("Pb_new");
    updateG();
    Fa = H_core + Ga;
    Fb = H_core + Gb;
  }
  if (k == max_iter)
  {
    cout << "Error: the job could not be finished in " << max_iter << "iterations.\n";
    return 1;
  }
  // Ea.print("Ea");
  // Eb.print("Eb");
  // Ca.print("Ca");
  // Cb.print("Cb");
  return 0;
}

double CNDO::getEnergy()
{
  arma::mat Ptotal = Pa + Pb;
  Ee = arma::dot(Pa, Ga) / 2. + arma::dot(Pb, Gb) / 2.;
  Ee += arma::dot(Ptotal, H_core);
  Etotal = Ee + Ec;

  // I dont need to keep printing these lines so let's get rid of them 
  //cout << std::setprecision(10);
  //cout << "Nuclear Repulsion Energy is " << Ec << " eV." << endl;
  //cout << "Electron Energy is " << Ee << " eV." << endl;
  return Etotal;
}


arma::mat CNDO::getGradient(){
  int num_atoms = mol.getnum_atoms();
  arma::mat gradient= arma::zeros(3, num_atoms);
  arma::mat OV_RA(3, dim*dim);
  mol.eval_OV1stmat(OV_RA);

  Suv_RA = OV_RA; // WRITE TO PUBLIC ATTRIBUTE 

  // OV_RA.print("Suv_RA (A is the center of u)");
  arma::mat gamma_RA(3, num_atoms*num_atoms);
  mol.eval_gamma1stmat(gamma_RA);
  // gamma_RA.print("gammaAB_RA");

  gammaAB_RA = gamma_RA; // WRITE TO PUBLIC ATTRIBUTE
  
  arma::mat P = Pa + Pb;
  arma::vec P_t = arma::zeros(num_atoms);
  size_t k_AO = 0;
  for (size_t k = 0; k < num_atoms; k++)
    for (auto ao : mol.mAtoms[k].mAOs)
    {
      P_t(k) += P(k_AO, k_AO);
      k_AO++;
    }
  // P.print("P");

  size_t k_AO_off = 0;
  for (size_t k = 0; k < num_atoms; k++)
  {
    Atom &A_Atom = mol.mAtoms[k];
    double betaA =CNDO_para_map[A_Atom.name].beta;
    size_t j_AO_off = 0;
    arma::vec gradient_A(gradient.colptr(k), 3, false, true );
    for (size_t j = 0; j < num_atoms; j++)
    {
      Atom &B_Atom = mol.mAtoms[j];
      double betaApB =betaA + CNDO_para_map[B_Atom.name].beta;
      if(j == k){
        j_AO_off += B_Atom.mAOs.size();
        continue;
      }
      double temp = P_t(k) *P_t(j) - P_t(k)* B_Atom.VAN - P_t(j)* A_Atom.VAN;
      gradient_A += temp * gamma_RA.col(k * num_atoms + j);
      // gradient_A += P_t(k) *(P_t(j) - 2* B_Atom.VAN) * gamma_RA.col(k * num_atoms + j);
      for (size_t k_AO = k_AO_off; k_AO < k_AO_off + A_Atom.mAOs.size(); k_AO ++)
        for (size_t j_AO = j_AO_off; j_AO < j_AO_off + B_Atom.mAOs.size(); j_AO ++)
        {
            //     cout << " j =  " << j << " k =  " << k  << " j_AO =  " << j_AO << " k_AO =  " << k_AO <<" " <<betaApB * P(k_AO, j_AO) << endl;
            //     arma::vec temp = OV_RA.col(k * num_atoms + j);
            // temp.print("temp");
            gradient_A += betaApB * P(k_AO, j_AO) * OV_RA.col(k_AO * dim + j_AO);
            gradient_A -= (Pa(k_AO, j_AO) * Pa(k_AO, j_AO) + Pb(k_AO, j_AO) * Pb(k_AO, j_AO)) *gamma_RA.col(k * num_atoms + j);
        }
      j_AO_off += B_Atom.mAOs.size();
    }
    k_AO_off += A_Atom.mAOs.size();
  }
  if (k_AO != dim)
    cout << "warn! the number of AOs is wrong." << endl;

  arma::mat nuc = arma::zeros(3, num_atoms);
  mol.eval_nuc_1st(nuc);
  gradient_nuclear = nuc; // SAVE TO THE PUBLIC INTERFACE 
  // nuc.print("gradient (Nuclear part)");
  // gradient.print("gradient (Electron part)");


  return gradient + nuc;
}


arma::mat CNDO::getNumericalGradient(double epsilon) {
    int num_atoms = mol.getnum_atoms();
    numerical_gradient = arma::zeros(3, num_atoms);
    
    // Save original positions
    std::vector<arma::vec> original_positions(num_atoms);
    for (int a = 0; a < num_atoms; a++) {
        original_positions[a] = mol.mAtoms[a].mAOs[0].get_R0();
    }
    
    for (int a = 0; a < num_atoms; a++) {
        for (int d = 0; d < 3; d++) {
            // Forward perturbation
            arma::vec R0 = mol.mAtoms[a].mAOs[0].get_R0();
            R0(d) += epsilon;
            mol.mAtoms[a].set_R0(R0);
            
            CNDO forward_scf(mol, p, q, max_iter, tol);
            forward_scf.init();
            forward_scf.run();
            double forward_energy = forward_scf.getEnergy();
            
            // Backward perturbation
            R0(d) -= 2 * epsilon;
            mol.mAtoms[a].set_R0(R0);
            
            CNDO backward_scf(mol, p, q, max_iter, tol);
            backward_scf.init();
            backward_scf.run();
            double backward_energy = backward_scf.getEnergy();
            
            // Compute gradient using central difference
            numerical_gradient(d, a) = (forward_energy - backward_energy) / (2 * epsilon);
            
            // Restore original position
            mol.mAtoms[a].set_R0(original_positions[a]);
        }
    }
    
    return numerical_gradient;
}
