#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <memory>
#include <string>
#include <chrono>
#include <iostream>

#include <TChain.h>
#include <TFile.h>
#include <TTree.h>
#include <TGeoGlobalMagField.h>

#include <FairEventHeader.h>
#include <FairGeoParSet.h>
#include <FairLogger.h>

#include "SimulationDataFormat/MCEventHeader.h"

#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"

#include "Field/MagneticField.h"

#include "ITSBase/GeometryTGeo.h"

#include "ITStracking/ROframe.h"
#include "ITStracking/IOUtils.h"
#include "ITStracking/Tracker.h"
#include "ITStracking/TrackerTraitsCPU.h"

#include "MathUtils/Utils.h"

#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

// #include "GPU/GPUO2Interface.h"
// #include "GPU/GPUReconstruction.h"
// #include "GPU/GPUChainITS.h"

#include <TGraph.h>

#include "ITStracking/Constants.h"
#include "ITStracking/Configuration.h"

using namespace o2::gpu;
using o2::its::MemoryParameters;
using o2::its::TrackingParameters;

using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;
using MCLabCont = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

void run_trac_ca_its_cosmics(std::string path = "./",
                            std::string outputfile = "o2trac_its.root",
                            std::string inputClustersITS = "o2clus_its.root", std::string inputGeom = "O2geometry.root",
                            std::string inputGRP = "o2sim_grp.root", std::string simfilename = "o2sim.root",
                            std::string paramfilename = "o2sim_par.root")
{

  gSystem->Load("libO2ITStracking.so");

  // std::unique_ptr<GPUReconstruction> rec(GPUReconstruction::CreateInstance());
  //std::unique_ptr<GPUReconstruction> rec(GPUReconstruction::CreateInstance("CUDA", true)); // for GPU with CUDA
  // auto* chainITS = rec->AddChain<GPUChainITS>();
  // rec->Init();

  //o2::its::Tracker tracker(chainITS->GetITSTrackerTraits());
  o2::its::Tracker tracker(new o2::its::TrackerTraitsCPU());
  o2::its::ROframe event(0);

  if (path.back() != '/') {
    path += '/';
  }

  //-------- init geometry and field --------//
  const auto grp = o2::parameters::GRPObject::loadFrom(path + inputGRP);
  if (!grp) {
    LOG(FATAL) << "Cannot run w/o GRP object";
  }
  o2::base::GeometryManager::loadGeometry(path + inputGeom, "FAIRGeom");
  o2::base::Propagator::initFieldFromGRP(grp);
  auto field = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
  if (!field) {
    LOG(FATAL) << "Failed to load magnetic field";
  }
  double origD[3] = {0., 0., 0.};
  tracker.setBz(field->getBz(origD));
  std::cout<<"Magnetic field: "<<field->getBz(origD)<<std::endl;

  bool isITS = grp->isDetReadOut(o2::detectors::DetID::ITS);
  if (!isITS) {
    LOG(WARNING) << "ITS is not in the readout";
    return;
  }
  bool isContITS = grp->isDetContinuousReadOut(o2::detectors::DetID::ITS);
  LOG(INFO) << "ITS is in " << (isContITS ? "CONTINUOS" : "TRIGGERED") << " readout mode";

  auto gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L, o2::TransformType::T2GRot,
                                            o2::TransformType::L2G)); // request cached transforms


  //>>>---------- attach input data --------------->>>
  TChain itsClusters("o2sim");
  itsClusters.AddFile((path + inputClustersITS).data());

  //<<<---------- attach input data ---------------<<<
  if (!itsClusters.GetBranch("ITSCluster")) {
    LOG(FATAL) << "Did not find ITS clusters branch ITSCluster in the input tree";
  }
  std::vector<o2::itsmft::Cluster>* clusters = nullptr;
  itsClusters.SetBranchAddress("ITSCluster", &clusters);

  if (!itsClusters.GetBranch("ITSClusterMCTruth")) {
    LOG(FATAL) << "Did not find ITS clusters branch ITSClusterMCTruth in the input tree";
  }
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* labels = nullptr;
  itsClusters.SetBranchAddress("ITSClusterMCTruth", &labels);

  std::vector<o2::itsmft::MC2ROFRecord>* mc2rofs = nullptr;
  if (!itsClusters.GetBranch("ITSClustersMC2ROF")) {
    LOG(FATAL) << "Did not find ITS clusters branch ITSClustersROF in the input tree";
  }
  itsClusters.SetBranchAddress("ITSClustersMC2ROF", &mc2rofs);

  std::vector<o2::its::TrackITSExt> tracks;
  // create/attach output tree
  TFile outFile(outputfile  .data(), "recreate");
  TTree outTree("o2sim", "CA ITS Tracks");
  std::vector<o2::its::TrackITS> tracksITS, *tracksITSPtr = &tracksITS;
  std::vector<int> trackClIdx, *trackClIdxPtr = &trackClIdx;
  std::vector<o2::itsmft::ROFRecord> vertROFvec, *vertROFvecPtr = &vertROFvec;
  std::vector<Vertex> vertices, *verticesPtr = &vertices;

  MCLabCont trackLabels, *trackLabelsPtr = &trackLabels;
  outTree.Branch("ITSTrack", &tracksITSPtr);
  outTree.Branch("ITSTrackClusIdx", &trackClIdxPtr);
  outTree.Branch("ITSTrackMCTruth", &trackLabelsPtr);
  outTree.Branch("ITSTracksMC2ROF", &mc2rofs);
  outTree.Branch("Vertices", &verticesPtr);
  outTree.Branch("VerticesROF", &vertROFvecPtr);
  if (!itsClusters.GetBranch("ITSClustersROF")) {
    LOG(FATAL) << "Did not find ITS clusters branch ITSClustersROF in the input tree";
  }
  std::vector<o2::itsmft::ROFRecord>* rofs = nullptr;
  itsClusters.SetBranchAddress("ITSClustersROF", &rofs);
  itsClusters.GetEntry(0);

  int roFrameCounter{0};

  std::vector<double> ncls;
  std::vector<double> time;

  // const float kmaxDCAxy1[5] = /*{1.f,0.5,0.5,1.7,3.};/*/{1.f,0.4f,0.4f,1.5f,3.f};
  // const float kmaxDCAz1[5] = /*{2.f,0.8,0.8,3.,5.};/*/{1.f,0.4f,0.4f,1.5f,3.f};
  // const float kmaxDN1[4] = /*{0.006f,0.0045f,0.01f,0.04f};/*/{0.005f,0.0035f,0.009f,0.03f};
  // const float kmaxDP1[4] = /*{0.04f,0.01f,0.012f,0.014f};/*/{0.02f,0.005f,0.006f,0.007f};
  // const float kmaxDZ1[6] = /*{1.5f,1.5f,2.f,2.f,2.f,2.f};/*/{1.f,1.f,1.5f,1.5f,1.5f,1.5f};
  // const float kDoublTanL1 = /*0.12f;/*/0.05f;
  // const float kDoublPhi1 = /*0.4f;/*/0.2f;

  // std::vector<TrackingParameters> trackParams(2);
  // trackParams[1].MinTrackLength = 7;
  // trackParams[1].TrackletMaxDeltaPhi = 0.3;
  // trackParams[1].CellMaxDeltaPhi = 0.2;
  // trackParams[1].CellMaxDeltaTanLambda = 0.05;
  // std::copy(kmaxDZ1, kmaxDZ1 + 6, trackParams[1].TrackletMaxDeltaZ);
  // std::copy(kmaxDCAxy1,kmaxDCAxy1+5,trackParams[1].CellMaxDCA);
  // std::copy(kmaxDCAz1, kmaxDCAz1+5, trackParams[1].CellMaxDeltaZ);
  // std::copy(kmaxDP1, kmaxDP1+4, trackParams[1].NeighbourMaxDeltaCurvature);
  // std::copy(kmaxDN1, kmaxDN1+4, trackParams[1].NeighbourMaxDeltaN);

  // std::vector<MemoryParameters> memParams(2);
  // for (auto& coef : memParams[1].CellsMemoryCoefficients)
  //   coef *= 10;
  // for (auto& coef : memParams[1].TrackletsMemoryCoefficients)
  //   coef *= 10;


  std::vector<TrackingParameters> trackParams(1);
  std::vector<MemoryParameters> memParams(1);
  trackParams[0].MinTrackLength = 3;
  trackParams[0].TrackletMaxDeltaPhi = o2::its::constants::math::Pi * 0.5f;
  for (int iLayer = 0; iLayer < o2::its::constants::its::TrackletsPerRoad; iLayer++) {
    trackParams[0].TrackletMaxDeltaZ[iLayer] = o2::its::constants::its::LayersZCoordinate()[iLayer + 1];
    memParams[0].TrackletsMemoryCoefficients[iLayer] = 0.5f;
    // trackParams[0].TrackletMaxDeltaZ[iLayer] = 10.f;
  }
  for (int iLayer = 0; iLayer < o2::its::constants::its::CellsPerRoad; iLayer++) {
    trackParams[0].CellMaxDCA[iLayer] = 10000.f;  //cm
    trackParams[0].CellMaxDeltaZ[iLayer] = 10000.f;  //cm
    memParams[0].CellsMemoryCoefficients[iLayer] = 0.001f;
  }

  
  tracker.setParameters(memParams, trackParams);

  for (auto& rof : *rofs) {

    std::cout<<std::endl<<"PROCESSING ROF: "<<roFrameCounter<<std::endl<<std::endl;
    
    auto start = std::chrono::steady_clock::now();

    o2::its::ioutils::loadROFrameData(rof, event, gsl::span(clusters->data(), clusters->size()), labels);

    // define a dummy vertex (0,0,0)
    auto& vtxROF = vertROFvec.emplace_back(rof); // register entry and number of vertices in the
    vtxROF.setFirstEntry(vertices.size());       // dedicated ROFRecord
    vtxROF.setNEntries(1);
    Vertex dummyVtx = Vertex(Point3D<float>(0., 0., 0.), std::array<float, 6>{0., 0., 0., 0., 0., 0.}, 50, 0.);
    dummyVtx.setTimeStamp(event.getROFrameId());
    vertices.push_back(dummyVtx);

    std::cout << " - Dummy vertex: x = " << dummyVtx.getX() << " y = " << dummyVtx.getY() << " x = " << dummyVtx.getZ() << std::endl;
    event.addPrimaryVertex(dummyVtx.getX(), dummyVtx.getY(), dummyVtx.getZ());
    
    trackClIdx.clear();
    tracksITS.clear();
    tracker.clustersToTracks(event);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> diff_t{end - start};

    ncls.push_back(event.getTotalClusters());
    time.push_back(diff_t.count());

    tracks.swap(tracker.getTracks());
    for (auto& trc : tracks) {
      trc.setFirstClusterEntry(trackClIdx.size()); // before adding tracks, create final cluster indices
      int ncl = trc.getNumberOfClusters();
      for (int ic = 0; ic < ncl; ic++) {
        trackClIdx.push_back(trc.getClusterIndex(ic));
      }
      tracksITS.emplace_back(trc);
    }

    trackLabels = tracker.getTrackLabels(); /// FIXME: assignment ctor is not optimal.
    outTree.Fill();
    roFrameCounter++;
  }

  outFile.cd();
  outTree.Write();
  outFile.Close();

  TGraph* graph = new TGraph(ncls.size(), ncls.data(), time.data());
  graph->SetMarkerStyle(20);
  graph->Draw("AP");
}

#endif
