#include "stdafx.h"
#include "n2v.h"

void node2vec(PWNet& InNet, const double& ParamP, const double& ParamQ,
  const int& Dimensions, const int& WalkLen, const int& NumWalks,
  const int& WinSize, const int& Iter, const bool& Verbose,
  const TStr& OutRWFile,
  const bool& LearnEmbeddingFlag,
  TIntFltVH& EmbeddingsHV) {
  //Preprocess transition probabilities
  PreprocessTransitionProbs(InNet, ParamP, ParamQ, Verbose);
  TIntV NIdsV;
  for (TWNet::TNodeI NI = InNet->BegNI(); NI < InNet->EndNI(); NI++) {
    NIdsV.Add(NI.GetId());
  }
  //Generate random walks
  int64 AllWalks = (int64)NumWalks * NIdsV.Len();
//  TVVec<TInt, int64> WalksVV;
//  WalksVV = TVVec<TInt, int64>(AllWalks,WalkLen);
  TRnd Rnd(time(NULL));
  int64 WalksDone = 0;
  TFOut FOut(OutRWFile);
  for (int64 i = 0; i < NumWalks; i++) {
    NIdsV.Shuffle(Rnd);
#pragma omp parallel for schedule(dynamic)
    for (int64 j = 0; j < NIdsV.Len(); j++) {
//      TInt nid = NIdsV[j];
      if ( Verbose && WalksDone%10000 == 0 ) {
        printf("\rWalking Progress: %.2lf%%",(double)WalksDone*100/(double)AllWalks);fflush(stdout);
      }
      TIntV WalkV;
      SimulateWalk(InNet, NIdsV[j], WalkLen, Rnd, WalkV);
#pragma omp critical (RWFileOut)
      {
        for (int64 k = 0; k < WalkV.Len(); k++) {
//          WalksVV.PutXY(i * NIdsV.Len() + j, k, WalkV[k]);
          FOut.PutInt(WalkV[k]);
          if (k + 1 == WalkV.Len()) {
            FOut.PutLn();
          } else {
            FOut.PutCh(' ');
          }
        }
      }
      WalksDone++;
    }
  }
  printf("\rRandom Walking Done.\n");
  fflush(stdout);

  if (Verbose) {
    printf("\n");
    fflush(stdout);
  }
  //Learning embeddings
  //READ Random Walked File
  if (LearnEmbeddingFlag) {
    TVVec <TInt, int64> WalksVV;
    WalksVV = TVVec<TInt, int64>(AllWalks, WalkLen);
    ReadRandomWalkedFile(OutRWFile, WalksVV);
    LearnEmbeddings(WalksVV, Dimensions, WinSize, Iter, Verbose, EmbeddingsHV);
  }
}

void node2vec(PWNet& InNet, const double& ParamP, const double& ParamQ,
  const int& Dimensions, const int& WalkLen, const int& NumWalks,
  const int& WinSize, const int& Iter, const bool& Verbose,
  TIntFltVH& EmbeddingsHV) {
//  TVVec <TInt, int64> WalksVV;
  TStr OutRWFile = "randomwalk.txt";
  bool LearnEmbeddingFlag = 1;
  node2vec(InNet, ParamP, ParamQ, Dimensions, WalkLen, NumWalks, WinSize,
   Iter, Verbose, OutRWFile, LearnEmbeddingFlag, EmbeddingsHV);
}


void node2vec(const PNGraph& InNet, const double& ParamP, const double& ParamQ,
  const int& Dimensions, const int& WalkLen, const int& NumWalks,
  const int& WinSize, const int& Iter, const bool& Verbose,
  const TStr& OutRWFile,
  TIntFltVH& EmbeddingsHV) {
  PWNet NewNet = PWNet::New();
  for (TNGraph::TEdgeI EI = InNet->BegEI(); EI < InNet->EndEI(); EI++) {
    if (!NewNet->IsNode(EI.GetSrcNId())) { NewNet->AddNode(EI.GetSrcNId()); }
    if (!NewNet->IsNode(EI.GetDstNId())) { NewNet->AddNode(EI.GetDstNId()); }
    NewNet->AddEdge(EI.GetSrcNId(), EI.GetDstNId(), 1.0);
  }
  bool LearnEmbeddingFlag = 1;
  node2vec(NewNet, ParamP, ParamQ, Dimensions, WalkLen, NumWalks, WinSize,
           Iter, Verbose, OutRWFile, LearnEmbeddingFlag, EmbeddingsHV);
}

void node2vec(const PNGraph& InNet, const double& ParamP, const double& ParamQ,
  const int& Dimensions, const int& WalkLen, const int& NumWalks,
  const int& WinSize, const int& Iter, const bool& Verbose,
  TIntFltVH& EmbeddingsHV) {
//  TVVec <TInt, int64> WalksVV;
  TStr OutRWFile = "randomwalk.txt";
  node2vec(InNet, ParamP, ParamQ, Dimensions, WalkLen, NumWalks, WinSize,
           Iter, Verbose, OutRWFile, EmbeddingsHV);
}

void node2vec(const PNEANet& InNet, const double& ParamP, const double& ParamQ,
  const int& Dimensions, const int& WalkLen, const int& NumWalks,
  const int& WinSize, const int& Iter, const bool& Verbose,
  const TStr& OutRWFile,
  TIntFltVH& EmbeddingsHV) {
  PWNet NewNet = PWNet::New();
  for (TNEANet::TEdgeI EI = InNet->BegEI(); EI < InNet->EndEI(); EI++) {
    if (!NewNet->IsNode(EI.GetSrcNId())) { NewNet->AddNode(EI.GetSrcNId()); }
    if (!NewNet->IsNode(EI.GetDstNId())) { NewNet->AddNode(EI.GetDstNId()); }
    NewNet->AddEdge(EI.GetSrcNId(), EI.GetDstNId(), InNet->GetFltAttrDatE(EI,"weight"));
  }
  bool LearnEmbeddingFlag = 1;
  node2vec(NewNet, ParamP, ParamQ, Dimensions, WalkLen, NumWalks, WinSize,
           Iter, Verbose, OutRWFile, LearnEmbeddingFlag, EmbeddingsHV);
}

void node2vec(const PNEANet& InNet, const double& ParamP, const double& ParamQ,
  const int& Dimensions, const int& WalkLen, const int& NumWalks,
  const int& WinSize, const int& Iter, const bool& Verbose,
 TIntFltVH& EmbeddingsHV) {
//  TVVec <TInt, int64> WalksVV;
  TStr OutRWFile = "randomwalk.txt";
  node2vec(InNet, ParamP, ParamQ, Dimensions, WalkLen, NumWalks, WinSize,
           Iter, Verbose, OutRWFile, EmbeddingsHV);
}


void ReadRandomWalkedFile(const TStr& InRWFile, TVVec<TInt, int64>& WalksVV) {
  TFIn FIn(InRWFile);
  int64 LineCnt = 0;
  while (!FIn.Eof()) {
    TStr Ln;
    FIn.GetNextLn(Ln);
    TStr Line, Comment;
    Ln.SplitOnCh(Line,'#',Comment);
    TStrV Tokens;
    Line.SplitOnWs(Tokens);
    TIntV WalkV;
    for (int64 i = 0; i < Tokens.Len(); i++) {
      TInt NId = Tokens[i].GetInt();
      WalksVV.PutXY(LineCnt, i, NId);
    }
    LineCnt++;
  }
}
