
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_EBMultiFabUtil.H>
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>


#include <AMReX_EB2_IF_Union.H>
#include <AMReX_EB2_IF_Intersection.H>
#include <AMReX_EB2_IF_Complement.H>
#include <AMReX_EB2_IF_Scale.H>
#include <AMReX_EB2_IF_Translation.H>
#include <AMReX_EB2_IF_Lathe.H>
#include <AMReX_EB2_IF_Box.H>
#include <AMReX_EB2_IF_Cylinder.H>
#include <AMReX_EB2_IF_Ellipsoid.H>
#include <AMReX_EB2_IF_Sphere.H>
#include <AMReX_EB2_IF_Plane.H>

#include <AMReX_ParmParse.H>
#include <AMReX_MLEBABecLap.H>
#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_MLMG.H>
#include <AMReX_GMRES_MLMG.H>

#include "Poisson.H"

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    {
        int verbose = 1;
        int n_cell = 128;
        int max_grid_size = 32;
        int use_GMRES = 0;
        amrex::Vector<int> n_cell_2d;
        // read parameters
        {
            ParmParse pp;
            pp.query("verbose", verbose);
        //    pp.query("n_cell", n_cell);
            pp.queryarr("n_cell", n_cell_2d);
            pp.query("max_grid_size", max_grid_size);
            pp.query("use_GMRES",use_GMRES);
        }

        Geometry geom;
        BoxArray grids;
        DistributionMapping dmap;
        {
            RealBox rb({AMREX_D_DECL(-.1035,-0.0527,-0.0527)}, {AMREX_D_DECL(0.1035,0.0527,0.0527)});
            Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1,1,1)};
            Box domain(IntVect{AMREX_D_DECL(0,0,0)},
                       IntVect{AMREX_D_DECL(n_cell_2d[0]-1,n_cell_2d[1]-1,0)});
            geom.define(domain, rb, CoordSys::cartesian, is_periodic);

            grids.define(domain); // define the BoxArray to be a single grid
            grids.maxSize(max_grid_size); // chop domain up into boxes with length max_Grid_size

            dmap.define(grids); // create a processor distribution mapping given the BoxARray
        }

        int required_coarsening_level = 0; // typically the same as the max AMR level index
        int max_coarsening_level = 100;    // typically a huge number so MG coarsens as much as possible
        // build a simple geometry using the "eb2." parameters in the inputs file
        ParmParse pp("eb2");
        std::string geom_type;
        pp.get("geom_type", geom_type);
        if (geom_type == "merge") {
            EB2::BoxIF box1({AMREX_D_DECL(-0.052,-0.0527,0)},
                           {AMREX_D_DECL(0.052, -0.0128,0)},false);
            EB2::BoxIF box2({AMREX_D_DECL(-0.052,0.0128,0)},
                            {AMREX_D_DECL(0.052,0.0527  ,0)},false);
            auto twoboxes = EB2::makeUnion(box1,box2);
            auto gshop = EB2::makeShop(twoboxes);
            EB2::Build(gshop, geom, required_coarsening_level, max_coarsening_level);
        } else {
            EB2::Build(geom, required_coarsening_level, max_coarsening_level);
        }

        const EB2::IndexSpace& eb_is = EB2::IndexSpace::top();
        const EB2::Level& eb_level = eb_is.getLevel(geom);

        // options are basic, volume, or full
        EBSupport ebs = EBSupport::full;

        // number of ghost cells for each of the 3 EBSupport types
        Vector<int> ng_ebs = {2,2,2};

        // This object provides access to the EB database in the format of basic AMReX objects
        // such as BaseFab, FArrayBox, FabArray, and MultiFab
        EBFArrayBoxFactory factory(eb_level, geom, grids, dmap, ng_ebs, ebs);

        // charge density and electric potential are nodal
        const BoxArray& nba = amrex::convert(grids, IntVect::TheNodeVector());
        MultiFab q  (nba, dmap, 1, 0, MFInfo(), factory);
        MultiFab phi(nba, dmap, 1, 0, MFInfo(), factory);

        InitData(q, n_cell_2d[0]/2, n_cell_2d[1]/2);

        LPInfo info;

//        MLEBABecLap mlebabec({geom},{grids},{dmap},info,{&factory});
        MLEBNodeFDLaplacian linop({geom},{grids},{dmap},info,{&factory});

        // define array of LinOpBCType for domain boundary conditions
        std::array<LinOpBCType,AMREX_SPACEDIM> bc_lo;
        std::array<LinOpBCType,AMREX_SPACEDIM> bc_hi;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            bc_lo[idim] = LinOpBCType::Periodic;
            bc_hi[idim] = LinOpBCType::Periodic;
        }

        // Boundary of the whole domain. This functions must be called,
        // and must be called before other bc functions.
        linop.setDomainBC(bc_lo,bc_hi);

        // see AMReX_MLLinOp.H for an explanation
        linop.setLevelBC(0, nullptr);

        //// operator looks like (ACoef - div BCoef grad) phi = rhs

        //// set ACoef to zero
        //MultiFab acoef(grids, dmap, 1, 0, MFInfo(), factory);
        //acoef.setVal(0.);
        //mlebabec.setACoeffs(0, acoef);

        //// set BCoef to 1.0 (and array of face-centered coefficients)
        //Array<MultiFab,AMREX_SPACEDIM> bcoef;
        //for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        //    bcoef[idim].define(amrex::convert(grids,IntVect::TheDimensionVector(idim)), dmap, 1, 0, MFInfo(), factory);
        //    bcoef[idim].setVal(1.0);
        //}
        //mlebabec.setBCoeffs(0, amrex::GetArrOfConstPtrs(bcoef));

        //// scaling factors; these multiply ACoef and BCoef
        //Real ascalar = 0.0;
        //Real bscalar = 1.0;
        //mlebabec.setScalars(ascalar, bscalar);

        //// think of this beta as the "BCoef" associated with an EB face
        //MultiFab beta(nba, dmap, 1, 0, MFInfo(), factory);
        //beta.setVal(1.);

        // set homogeneous Dirichlet BC for EB
        linop.setEBDirichlet(0.);

        MLMG mlmg(linop);

        // relative and absolute tolerances for linear solve
        const Real tol_rel = 1.e-10;
        const Real tol_abs = 0.0;

        mlmg.setVerbose(verbose);
        // Solve linear system
        phi.setVal(0.0); // initial guess for phi

        if (use_GMRES) {
            amrex::GMRESMLMG gmsolve(mlmg);
            gmsolve.setVerbose(verbose);
            gmsolve.solve(phi, q, tol_rel, tol_abs);
            amrex::Vector<amrex::MultiFab> vmf;
            vmf.emplace_back(phi, amrex::make_alias, 0, phi.nComp());
            linop.postSolve(vmf);
        } else {
            mlmg.solve({&phi}, {&q}, tol_rel, tol_abs);
        }

        //// store plotfile variables; q and phi
        MultiFab plotfile_mf(grids, dmap, 2, 0, MFInfo(), factory);
        amrex::average_node_to_cellcenter(plotfile_mf, 0, q, 0, 1);
        amrex::average_node_to_cellcenter(plotfile_mf, 1, phi, 0, 1);
        //MultiFab::Copy(plotfile_mf,  q,0,0,1,0);
        //MultiFab::Copy(plotfile_mf,phi,0,1,1,0);

        EB_WriteSingleLevelPlotfile("plt", plotfile_mf, {"q", "phi"}, geom, 0.0, 0);
    }

    amrex::Finalize();
}
