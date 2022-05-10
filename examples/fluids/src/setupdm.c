// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Setup DM for Navier-Stokes example using PETSc

#include "../navierstokes.h"

// Create mesh
PetscErrorCode CreateDM(MPI_Comm comm, ProblemData *problem, DM *dm) {
  PetscErrorCode   ierr;
  PetscFunctionBeginUser;
  // Create DMPLEX
  ierr = DMCreate(comm, dm); CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX); CHKERRQ(ierr);
  // Set Tensor elements
  ierr = PetscOptionsSetValue(NULL, "-dm_plex_simplex", "0"); CHKERRQ(ierr);
  // Set CL options
  ierr = DMSetFromOptions(*dm); CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view"); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// Setup DM
PetscErrorCode SetUpDM(DM dm, ProblemData *problem, PetscInt degree,
                       SimpleBC bc, Physics phys) {
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  {
    // Configure the finite element space and boundary conditions
    PetscFE  fe;
    PetscInt num_comp_q = 5;
    DMLabel label;
    ierr = PetscFECreateLagrange(PETSC_COMM_SELF, problem->dim, num_comp_q,
                                 PETSC_FALSE, degree, PETSC_DECIDE,
                                 &fe); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)fe, "Q"); CHKERRQ(ierr);
    ierr = DMAddField(dm, NULL,(PetscObject)fe); CHKERRQ(ierr);
    ierr = DMCreateDS(dm); CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "Face Sets", &label); CHKERRQ(ierr);
    // Set wall BCs
    if (bc->num_wall > 0) {
      ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label,
                           bc->num_wall, bc->walls, 0, bc->num_comps,
                           bc->wall_comps, (void(*)(void))problem->bc,
                           NULL, problem->bc_ctx, NULL);  CHKERRQ(ierr);
    }
    // Set slip BCs in the x direction
    if (bc->num_slip[0] > 0) {
      PetscInt comps[1] = {1};
      ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipx", label,
                           bc->num_slip[0], bc->slips[0], 0, 1, comps,
                           (void(*)(void))NULL, NULL, problem->bc_ctx, NULL); CHKERRQ(ierr);
    }
    // Set slip BCs in the y direction
    if (bc->num_slip[1] > 0) {
      PetscInt comps[1] = {2};
      ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipy", label,
                           bc->num_slip[1], bc->slips[1], 0, 1, comps,
                           (void(*)(void))NULL, NULL, problem->bc_ctx, NULL); CHKERRQ(ierr);
    }
    // Set slip BCs in the z direction
    if (bc->num_slip[2] > 0) {
      PetscInt comps[1] = {3};
      ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipz", label,
                           bc->num_slip[2], bc->slips[2], 0, 1, comps,
                           (void(*)(void))NULL, NULL, problem->bc_ctx, NULL); CHKERRQ(ierr);
    }
    ierr = DMPlexSetClosurePermutationTensor(dm, PETSC_DETERMINE, NULL);
    CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe); CHKERRQ(ierr);
  }
  {
    // Empty name for conserved field (because there is only one field)
    PetscSection section;
    ierr = DMGetLocalSection(dm, &section); CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(section, 0, ""); CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 0, "Density");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 1, "Momentum X");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 2, "Momentum Y");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 3, "Momentum Z");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 4, "Energy Density");
    CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

// Refine DM for high-order viz
PetscErrorCode VizRefineDM(DM dm, User user, ProblemData *problem,
                           SimpleBC bc, Physics phys) {
  PetscErrorCode ierr;
  DM             dm_hierarchy[user->app_ctx->viz_refine + 1];
  VecType        vec_type;
  PetscFunctionBeginUser;

  ierr = DMPlexSetRefinementUniform(dm, PETSC_TRUE); CHKERRQ(ierr);

  dm_hierarchy[0] = dm;
  for (PetscInt i = 0, d = user->app_ctx->degree;
       i < user->app_ctx->viz_refine; i++) {
    Mat interp_next;
    ierr = DMRefine(dm_hierarchy[i], MPI_COMM_NULL, &dm_hierarchy[i+1]);
    CHKERRQ(ierr);
    ierr = DMClearDS(dm_hierarchy[i+1]); CHKERRQ(ierr);
    ierr = DMClearFields(dm_hierarchy[i+1]); CHKERRQ(ierr);
    ierr = DMSetCoarseDM(dm_hierarchy[i+1], dm_hierarchy[i]); CHKERRQ(ierr);
    d = (d + 1) / 2;
    if (i + 1 == user->app_ctx->viz_refine) d = 1;
    ierr = DMGetVecType(dm, &vec_type); CHKERRQ(ierr);
    ierr = DMSetVecType(dm_hierarchy[i+1], vec_type); CHKERRQ(ierr);
    ierr = SetUpDM(dm_hierarchy[i+1], problem, d, bc, phys);
    CHKERRQ(ierr);
    ierr = DMCreateInterpolation(dm_hierarchy[i], dm_hierarchy[i+1], &interp_next,
                                 NULL); CHKERRQ(ierr);
    if (!i) user->interp_viz = interp_next;
    else {
      Mat C;
      ierr = MatMatMult(interp_next, user->interp_viz, MAT_INITIAL_MATRIX,
                        PETSC_DECIDE, &C); CHKERRQ(ierr);
      ierr = MatDestroy(&interp_next); CHKERRQ(ierr);
      ierr = MatDestroy(&user->interp_viz); CHKERRQ(ierr);
      user->interp_viz = C;
    }
  }
  for (PetscInt i=1; i<user->app_ctx->viz_refine; i++) {
    ierr = DMDestroy(&dm_hierarchy[i]); CHKERRQ(ierr);
  }
  user->dm_viz = dm_hierarchy[user->app_ctx->viz_refine];

  PetscFunctionReturn(0);
}
