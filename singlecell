##26.07.2022
##ccRCC
library(Seurat)
library(tidyverse)
library(patchwork)
rm(list = ls())

##=== Create a list of Seurat objects and load expression matrices ===##
dir <- dir('~/GSE156632')  ##  target directory
dir <- paste0('~/GSE156632/', dir)  ##  Prepend sample names to file paths
dir 

#Assign sample names in the order of the files.
#(Names should not begin with a number and must not contain spaces.)
group = c('T','N','T','N','T','N','T','N','T','N')
samples_name = c('T1','N1','T2','N2','T3','N3','T4','N4','T5','N5')

scRNAlist <- list()  
for(i in 1:length(dir)){
  counts <- read.table(file = dir[i], row.names=1, sep = ',', header = T)
}
  unique(counts$Symbol) -> item  
  match(item, counts$Symbol) ->idx
  counts[idx,] -> counts
  rownames(counts) <- counts$Symbol  
  counts <- counts[,-1]

  scRNAlist[[i]] <- CreateSeuratObject(counts, project=samples_name[i],
                                       min.cells=3, min.features = 200)
  scRNAlist[[i]] <- RenameCells(scRNAlist[[i]], add.cell.id = samples_name[i]) #Add a prefix to each cell barcode
  if(T){    
+     scRNAlist[[i]][["percent.mt"]] <- PercentageFeatureSet(scRNAlist[[i]], pattern = "^MT-") 
+ }
 if(T){
+     scRNAlist[[i]][["percent.rb"]] <- PercentageFeatureSet(scRNAlist[[i]], pattern = "^RP[SL]")
+ }
VlnPlot(scRNAlist[[10]],features=c("nFeature_RNA","nCount_RNA","percent.mt","percent.rb"),ncol=4,pt.size=0.1)
#Filter out cells with high mitochondrial gene expression and outliers
plot1<-FeatureScatter(scRNAlist[[10]],feature1="nCount_RNA",feature2="percent.mt")
plot2<-FeatureScatter(scRNAlist[[10]],feature1="nCount_RNA",feature2="percent.rb")
plot1+plot2
plot3<-FeatureScatter(scRNAlist[[10]],feature1="nCount_RNA",feature2="nFeature_RNA")
names(scRNAlist) <- samples_name
dir.create("~/res_out/QC")  
save(scRNAlist, file = "~/res_out/QC/scRNAlist0.Rdata")

##=== Quality control for the single-cell RNA-seq expression matrix ===##
scRNA <- merge(scRNAlist[[1]], scRNAlist[2:length(scRNAlist)])  ## Merge multiple Seurat objects into one
scRNA$proj <- rep("10x", ncol(scRNA))  ## Annotate the data source ("10x") for all cells
# scRNAlist <- SplitObject(scRNA, split.by = "orig.ident")  ##  back into individual samples
head(scRNA@meta.data) 
##violin plot
theme.set1 = theme(axis.title.x=element_blank(), 
                   axis.text.x=element_blank(), 
                   axis.ticks.x=element_blank())
theme.set2 = theme(axis.title.x=element_blank())
plot.featrures = c("nFeature_RNA", "nCount_RNA", "percent.mt", "percent.rb")
group = "orig.ident"
#before filtering
plots = list()
for(i in seq_along(plot.featrures)){
  plots[[i]] = VlnPlot(scRNA, group.by=group, pt.size = 0,
               features = plot.featrures[i]) + theme.set2 + NoLegend()}
violin <- wrap_plots(plots = plots, nrow=2)    
ggsave("~/res_out/QC/vlnplot_before_qc.pdf", plot = violin, width = 26, height = 8) 
ggsave("~/res_out/QC/vlnplot_before_qc.png", plot = violin, width = 26, height = 8)  
#quality control thresholds
minGene=500
maxGene=4000
pctMT=15
pctRB=20
scRNA <- subset(scRNA, subset = nFeature_RNA > minGene & nFeature_RNA < 
                  maxGene & percent.mt < pctMT & percent.rb < pctRB)
#plot after filtering
plots = list()
for(i in seq_along(plot.featrures)){
  plots[[i]] = VlnPlot(scRNA, group.by=group, pt.size = 0,
               features = plot.featrures[i]) + theme.set2 + NoLegend()}
violin <- wrap_plots(plots = plots, nrow=2)     
ggsave("~/res_out/QC/vlnplot_after_qc.pdf", plot = violin, width = 26, height = 8) 

##=== Assigning cell cycle scores to single cells ===##
scRNA <- NormalizeData(scRNA) %>% FindVariableFeatures() %>%
         ScaleData(features = rownames(scRNA))
		 
g2m_genes <- cc.genes$g2m.genes
g2m_genes <- CaseMatch(search=g2m_genes, match=rownames(scRNA))
s_genes <- cc.genes$s.genes    
s_genes <- CaseMatch(search=s_genes, match=rownames(scRNA))
scRNA <- CellCycleScoring(scRNA, g2m.features=g2m_genes, s.features=s_genes)
tmp <- RunPCA(scRNA, features = c(g2m_genes, s_genes), verbose = F)
p <- DimPlot(tmp, reduction = "pca", group.by = "orig.ident")
ggsave("~/res_out/QC/CellCycle_pca.png", p, width = 8, height = 6)
rm(tmp)	

##=== Plotting the effects of mitochondrial and ribosomal gene expression ===##
## Assess the influence of mitochondrial genes on sample quality
mt.genes <- grep("^MT-", rownames(scRNA), value=T, ignore.case=T)
tmp <- RunPCA(scRNA, features = mt.genes, verbose = F)
p <- DimPlot(tmp, reduction = "pca", group.by = "orig.ident")
ggsave("~/res_out/QC/mito_pca.png", p, width = 8, height = 6)
rm(tmp)
##ribosomal genes
rb.genes <- grep("^RP[SL]", rownames(scRNA), value=T, ignore.case=T)
tmp <- RunPCA(scRNA, features = rb.genes, verbose = F)
p <- DimPlot(tmp, reduction = "pca", group.by = "orig.ident")
ggsave("~/res_out/QC/ribo_pca.png", p, width = 8, height = 6)
rm(tmp)
save(scRNA, file = "~/res_out/QC/scRNA_QC.Rdata")

##===batch effect==##
library(Seurat)
library(tidyverse)
library(SingleR)
library(harmony)
library(patchwork)
dir.create("Overview")
rm(list=ls())
set.seed(123456)

load("scRNA_QC.Rdata")
scRNA <- RunPCA(scRNA, verbose = F) ##reduce dimension
##Identify highly variable genes across cells
scRNA <- FindVariableFeatures(scRNA, selection.method = 'vst',nfeatures = 1500)
top10 <- head(VariableFeatures(scRNA), 10)
plot1 <- VariableFeaturePlot(scRNA)  
plot2 <- LabelPoints(plot = plot1, points = top10, repel = TRUE) 
p <- plot1+plot2
ggsave(filename = 'Overview/top10_highVarible_genes.png',p, width = 18, height = 6)

p <- ElbowPlot(scRNA, ndims = 50)
ggsave(filename = 'Overview/elbowpoints.png',p)
pc.num=1:15 #main PC
scRNA <- scRNA %>%  RunUMAP(dims=pc.num) %>%
    FindNeighbors(dims = pc.num) %>% FindClusters(resolution=0.8)
p1 <- DimPlot(scRNA, reduction = "umap", label = T) + NoLegend()
ggsave("Overview/UMAP_overview.png", p1, width = 8, height = 6)
head(scRNA@meta.data)
grep('T',scRNA@meta.data$orig.ident) ->idx_T
grep('N',scRNA@meta.data$orig.ident) ->idx_N
scRNA@meta.data$group <- 'NA'
scRNA@meta.data$group[idx_T] <- 'T'
scRNA@meta.data$group[idx_N] <- 'N'
table(scRNA@meta.data$orig.ident,scRNA@meta.data$group)

p1 <- DimPlot(scRNA, reduction = "umap", group.by = "orig.ident") 
p2 <- DimPlot(scRNA, reduction = 'umap', group.by = 'group')
pc = p1|p2
ggsave("Overview/batch_before_overview.png", pc, width = 15, height = 7)

theme.set2 = theme(axis.title.x=element_blank())
plot.featrures = c("S.Score", "G2M.Score")
plots = list()
for(i in seq_along(plot.featrures)){
    plots[[i]] = VlnPlot(scRNA, group.by="seurat_clusters", pt.size = 0,
                         features = plot.featrures[i]) + theme.set2 + NoLegend()}
violin <- wrap_plots(plots = plots, nrow=2)    
ggsave("Overview/CellCycle_violin.png", plot = violin, width = 8, height = 6) 

p1 <- DimPlot(scRNA, reduction = "umap", group.by = "Phase")
pc = p1 +  plot_layout(guides = "collect")
ggsave("Overview/CellCycle_dimplot.png", pc, width = 8, height = 4)
save(scRNA, file = "Overview/scRNA_overview.Rdata")

##=== SCT normalization and batch effect correction ===##
dir.create("Harmony")
rm(list=ls())
load("scRNA_QC.Rdata")
set.seed(123456)
scRNA <- SCTransform(scRNA, return.only.var.genes = F, 
                       vars.to.regress = c("S.Score", "G2M.Score"))
scRNA <- RunPCA(scRNA,  verbose=FALSE)
head(scRNA@meta.data)
grep('T',scRNA@meta.data$orig.ident) ->idx_T
grep('N',scRNA@meta.data$orig.ident) ->idx_N
scRNA@meta.data$group <- 'NA'
scRNA@meta.data$group[idx_T] <- 'T'
scRNA@meta.data$group[idx_N] <- 'N'
table(scRNA@meta.data$orig.ident,scRNA@meta.data$group)
scRNA <- RunHarmony(scRNA, group.by.vars="group", 
                      assay.use="SCT" )
p <- ElbowPlot(scRNA, ndims = 50)
ggsave(filename = 'Harmony/elbowplot2.png',p)
pc.num=1:25
