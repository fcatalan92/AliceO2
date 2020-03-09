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
#include "DetectorsCommonDataFormats/NameConf.h"

#include "SimulationDataFormat/MCEventHeader.h"

#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DetectorsCommonDataFormats/DetID.h"
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

#include "GPU/GPUO2Interface.h"
#include "GPU/GPUReconstruction.h"
#include "GPU/GPUChainITS.h"

#include <TGraph.h>

// #include "ITStracking/Constants.h"
#include "ITStracking/Configuration.h"

using namespace o2::gpu;
using o2::its::MemoryParameters;
using o2::its::TrackingParameters;

using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;
using MCLabCont = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

void run_trac_ca_its_cosmics(std::string path = "./",
                            std::string outputfile = "o2trac_its.root",
                            std::string inputClustersITS = "o2clus_its.root",
                            std::string dictfile = "",
                            std::string inputGRP = "o2sim_grp.root")
{

  gSystem->Load("libO2ITStracking.so");

  // std::unique_ptr<GPUReconstruction> rec(GPUReconstruction::CreateInstance());
  // std::unique_ptr<GPUReconstruction> rec(GPUReconstruction::CreateInstance("CUDA", true)); // for GPU with CUDA
  // auto* chainITS = rec->AddChain<GPUChainITS>();
  // rec->Init();

  // o2::its::Tracker tracker(chainITS->GetITSTrackerTraits());
  o2::its::Tracker tracker(new o2::its::TrackerTraitsCPU());
  o2::its::ROframe event(0, 7);

  if (path.back() != '/') {
    path += '/';
  }

  //-------- init geometry and field --------//
  const auto grp = o2::parameters::GRPObject::loadFrom(path + inputGRP);
  if (!grp) {
    LOG(FATAL) << "Cannot run w/o GRP object";
  }
  o2::base::GeometryManager::loadGeometry(path);
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

  if (!itsClusters.GetBranch("ITSClusterComp")) {
    LOG(FATAL) << "Did not find ITS clusters branch ITSClusterComp in the input tree";
  }
  std::vector<o2::itsmft::CompClusterExt>* cclusters = nullptr;
  itsClusters.SetBranchAddress("ITSClusterComp", &cclusters);

  if (!itsClusters.GetBranch("ITSClusterPatt")) {
    LOG(FATAL) << "Did not find ITS cluster patterns branch ITSClusterPatt in the input tree";
  }
  std::vector<unsigned char>* patterns = nullptr;
  itsClusters.SetBranchAddress("ITSClusterPatt", &patterns);

  std::vector<o2::itsmft::ROFRecord>* rofs = nullptr;
  if (!itsClusters.GetBranch("ITSClustersROF")) {
    LOG(FATAL) << "Did not find ITS clusters branch ITSClustersROF in the input tree";
  }
  itsClusters.SetBranchAddress("ITSClustersROF", &rofs);

  MCLabCont* labels = nullptr;
  if (!itsClusters.GetBranch("ITSClusterMCTruth")) {
    LOG(INFO) << "Did not find ITS clusters branch ITSClusterMCTruth in the input tree";
  } else {
    itsClusters.SetBranchAddress("ITSClusterMCTruth", &labels);
  }

  std::vector<o2::itsmft::MC2ROFRecord>* mc2rofs = nullptr;
  if (!itsClusters.GetBranch("ITSClustersMC2ROF")) {
    LOG(INFO) << "Did not find ITS clusters branch ITSClustersMC2ROF in the input tree";
  } else {
    itsClusters.SetBranchAddress("ITSClustersMC2ROF", &mc2rofs);
  }

  itsClusters.GetEntry(0);

  //-------------------------------------------------

  o2::itsmft::TopologyDictionary dict;
  if (dictfile.empty()) {
    dictfile = o2::base::NameConf::getDictionaryFileName(o2::detectors::DetID::ITS, "", ".bin");
  }
  std::ifstream file(dictfile.c_str());
  if (file.good()) {
    LOG(INFO) << "Running with dictionary: " << dictfile.c_str();
    dict.readBinaryFile(dictfile);
  } else {
    LOG(INFO) << "Running without dictionary !";
  }

  //-------------------------------------------------

  std::vector<o2::its::TrackITSExt> tracks;
  // create/attach output tree
  TFile outFile((path + outputfile).data(), "recreate");
  TTree outTree("o2sim", "CA ITS Tracks");
  std::vector<o2::its::TrackITS> tracksITS, *tracksITSPtr = &tracksITS;
  std::vector<int> trackClIdx, *trackClIdxPtr = &trackClIdx;
  std::vector<o2::itsmft::ROFRecord> vertROFvec, *vertROFvecPtr = &vertROFvec;
  std::vector<Vertex> vertices, *verticesPtr = &vertices;

  std::vector<o2::MCCompLabel> trackLabels, *trackLabelsPtr = &trackLabels;
  outTree.Branch("ITSTrack", &tracksITSPtr);
  outTree.Branch("ITSTrackClusIdx", &trackClIdxPtr);
  if (labels) {
    outTree.Branch("ITSTrackMCTruth", &trackLabelsPtr);
  }
  if (mc2rofs) {
    outTree.Branch("ITSTracksMC2ROF", &mc2rofs);
  }
  outTree.Branch("Vertices", &verticesPtr);
  outTree.Branch("VerticesROF", &vertROFvecPtr);

  int roFrameCounter{0};

  std::vector<double> ncls;
  std::vector<double> time;

  std::vector<TrackingParameters> trackParams(1);
  std::vector<MemoryParameters> memParams(1);
  trackParams[0].MinTrackLength = 3;
  trackParams[0].TrackletMaxDeltaPhi = o2::its::constants::math::Pi * 0.5f;
  for (int iLayer = 0; iLayer < trackParams[0].TrackletsPerRoad(); iLayer++) {
    trackParams[0].TrackletMaxDeltaZ[iLayer] = trackParams[0].LayerZ[iLayer + 1];
    memParams[0].TrackletsMemoryCoefficients[iLayer] = 0.5f;
    // trackParams[0].TrackletMaxDeltaZ[iLayer] = 10.f;
  }
  for (int iLayer = 0; iLayer < trackParams[0].CellsPerRoad(); iLayer++) {
    trackParams[0].CellMaxDCA[iLayer] = 10000.f;  //cm
    trackParams[0].CellMaxDeltaZ[iLayer] = 10000.f;  //cm
    memParams[0].CellsMemoryCoefficients[iLayer] = 0.001f;
  }

  tracker.setParameters(memParams, trackParams);

  int currentEvent = -1;
  gsl::span<const unsigned char> patt(patterns->data(), patterns->size());
  auto pattIt = patt.begin();
  auto clSpan = gsl::span(cclusters->data(), cclusters->size());

  for (auto& rof : *rofs) {

    std::cout<<std::endl<<"PROCESSING ROF: "<<roFrameCounter<<std::endl<<std::endl;
    
    auto start = std::chrono::steady_clock::now();
    auto it = pattIt;
    o2::its::ioutils::loadROFrameData(rof, event, clSpan, pattIt, dict, labels);

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
