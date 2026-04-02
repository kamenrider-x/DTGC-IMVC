function fea = NormalizeFea(fea,row)

if ~exist('row','var')
    row = 1; 
end

if row
    nSmp = size(fea,1); 
    
    feaNorm = max(1e-14,full(sum(fea.^2,2)));
    
    fea = spdiags(feaNorm.^-.5,0,nSmp,nSmp)*fea;
else
    nSmp = size(fea,2); 
    
    feaNorm = max(1e-14,full(sum(fea.^2,1))');
    
    fea = fea*spdiags(feaNorm.^-.5,0,nSmp,nSmp);
end
            
return;