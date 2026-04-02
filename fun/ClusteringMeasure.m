function [ACC, MIhat, Purity] = ClusteringMeasure(Y, predY)

if size(Y,2) ~= 1
    Y = Y';
end;
if size(predY,2) ~= 1
    predY = predY';
end;

n = length(Y);

uY = unique(Y);
nclass = length(uY);
Y0 = zeros(n,1);
if nclass ~= max(Y)
    for i = 1:nclass
        Y0(Y == uY(i)) = i;
    end;
    Y = Y0;
end;

uY = unique(predY);
nclass = length(uY);
predY0 = zeros(n,1);
if nclass ~= max(predY)
    for i = 1:nclass
        predY0(predY == uY(i)) = i;
    end;
    predY = predY0;
end;

Lidx = unique(Y);        
classnum = length(Lidx);
predLidx = unique(predY);     
pred_classnum = length(predLidx);

correnum = 0; 
for ci = 1:pred_classnum
    incluster = Y(predY == predLidx(ci));
    
    inclunub = hist(incluster, 1:max(incluster)); 
    if isempty(inclunub) 
        inclunub=0;
    end;
    
    correnum = correnum + max(inclunub);
end;
Purity = correnum / length(predY); 

res = bestMap(Y, predY);
ACC = length(find(Y == res))/length(Y);

MIhat = MutualInfo(Y,res);

end

function [newL2, c] = bestMap(L1,L2)
L1 = L1(:);
L2 = L2(:);
if size(L1) ~= size(L2)
    error('L1 和 L2 的大小必须一致');
end
L1 = L1 - min(L1) + 1; 
L2 = L2 - min(L2) + 1; 

nClass = max(max(L1), max(L2));
G = zeros(nClass); 
for i=1:nClass
    for j=1:nClass
        G(i,j) = length(find(L1 == i & L2 == j));
    end
end

[c,t] = hungarian(-G);

newL2 = zeros(nClass,1);
for i=1:nClass
    newL2(L2 == i) = c(i);
end
end

function MIhat = MutualInfo(L1,L2)
L1 = L1(:);
L2 = L2(:);
if size(L1) ~= size(L2)
    error('L1 和 L2 的大小必须一致');
end
L1 = L1 - min(L1) + 1; 
L2 = L2 - min(L2) + 1; 

nClass = max(max(L1), max(L2));
G = zeros(nClass); 
for i=1:nClass
    for j=1:nClass
        G(i,j) = length(find(L1 == i & L2 == j))+eps; 
    end
end
sumG = sum(G(:)); 

P1 = sum(G,2);  P1 = P1/sumG; 
P2 = sum(G,1);  P2 = P2/sumG; 
H1 = sum(-P1.*log2(P1)); 
H2 = sum(-P2.*log2(P2)); 
P12 = G/sumG; 

PPP = P12./repmat(P2,nClass,1)./repmat(P1,1,nClass);
PPP(abs(PPP) < 1e-12) = 1; 

MI = sum(P12(:) .* log2(PPP(:))); 
MIhat = MI / max(H1,H2); 
MIhat = real(MIhat); 
end

function [C,T]=hungarian(A)
[m,n]=size(A);
if (m~=n)
    error('匈牙利算法：成本矩阵必须是方阵！');
end
orig=A;
A=hminired(A);
[A,C,U]=hminiass(A);
while (U(n+1))
    LR=zeros(1,n);
    LC=zeros(1,n);
    CH=zeros(1,n);
    RH=[zeros(1,n) -1];
    SLC=[];
    r=U(n+1);
    LR(r)=-1;
    SLR=r;
    while (1)
        if (A(r,n+1)~=0)
            l=-A(r,n+1);
            if (A(r,l)~=0 & RH(r)==0)
                RH(r)=RH(n+1);
                RH(n+1)=r;
                CH(r)=-A(r,l);
            end
        else
            if (RH(n+1)<=0)
                [A,CH,RH]=hmreduce(A,CH,RH,LC,LR,SLC,SLR);
            end
            r=RH(n+1);
            l=CH(r);
            CH(r)=-A(r,l);
            if (A(r,l)==0)
                RH(n+1)=RH(r);
                RH(r)=0;
            end
        end
        while (LC(l)~=0)
            if (RH(r)==0)
                if (RH(n+1)<=0)
                    [A,CH,RH]=hmreduce(A,CH,RH,LC,LR,SLC,SLR);
                end
                r=RH(n+1);
            end
            l=CH(r);
            CH(r)=-A(r,l);
            if(A(r,l)==0)
                RH(n+1)=RH(r);
                RH(r)=0;
            end
        end
        if (C(l)==0)
            [A,C,U]=hmflip(A,C,LC,LR,U,l,r);
            break;
        else
            LC(l)=r;
            SLC=[SLC l];
            r=C(l);
            LR(r)=l;
            SLR=[SLR r];
        end
    end
end
T=sum(orig(logical(sparse(C,1:size(orig,2),1))));
end

function A=hminired(A)
[m,n]=size(A);
colMin=min(A);
A=A-colMin(ones(n,1),:);
rowMin=min(A')';
A=A-rowMin(:,ones(1,n));
[i,j]=find(A==0);
A(1,n+1)=0;
for k=1:n
    cols=j(k==i)';
    A(k,[n+1 cols])=[-cols 0];
end
end

function [A,C,U]=hminiass(A)
[n,np1]=size(A);
C=zeros(1,n);
U=zeros(1,n+1);
LZ=zeros(1,n);
NZ=zeros(1,n);
for i=1:n
	lj=n+1;
	j=-A(i,lj);
	while (C(j)~=0)
		lj=j;
		j=-A(i,lj);
		if (j==0)
			break;
		end
	end
	if (j~=0)
		C(j)=i;
		A(i,lj)=A(i,j);
		NZ(i)=-A(i,j);
		LZ(i)=lj;
		A(i,j)=0;
	else
		lj=n+1;
		j=-A(i,lj);
		while (j~=0)
			r=C(j);
			lm=LZ(r);
			m=NZ(r);
			while (m~=0)
				if (C(m)==0)
					break;
				end
				lm=m;
				m=-A(r,lm);
			end
			if (m==0)
				lj=j;
				j=-A(i,lj);
			else
				A(r,lm)=-j;
				A(r,j)=A(r,m);
				NZ(r)=-A(r,m);
				LZ(r)=j;
				A(r,m)=0;
				C(m)=r;
				A(i,lj)=A(i,j);
				NZ(i)=-A(i,j);
				LZ(i)=lj;
				A(i,j)=0;
				C(j)=i;
				break;
			end
		end
	end
end
r=zeros(1,n);
rows=C(C~=0);
r(rows)=rows;
empty=find(r==0);
U=zeros(1,n+1);
U([n+1 empty])=[empty 0];
end

function [A,C,U]=hmflip(A,C,LC,LR,U,l,r)
n=size(A,1);
while (1)
    C(l)=r;
    m=find(A(r,:)==-l);
    A(r,m)=A(r,l);
    A(r,l)=0;
    if (LR(r)<0)
        U(n+1)=U(r);
        U(r)=0;
        return;
    else
        l=LR(r);
        A(r,l)=A(r,n+1);
        A(r,n+1)=-l;
        r=LC(l);
    end
end
end

function [A,CH,RH]=hmreduce(A,CH,RH,LC,LR,SLC,SLR)
n=size(A,1);
coveredRows=LR==0;
coveredCols=LC~=0;
r=find(~coveredRows);
c=find(~coveredCols);
m=min(min(A(r,c)));
A(r,c)=A(r,c)-m;
for j=c
    for i=SLR
        if (A(i,j)==0)
            if (RH(i)==0)
                RH(i)=RH(n+1);
                RH(n+1)=i;
                CH(i)=j;
            end
            row=A(i,:);
            colsInList=-row(row<0);
            if (length(colsInList)==0)
                l=n+1;
            else
                l=colsInList(row(colsInList)==0);
            end
            A(i,l)=-j;
        end
    end
end
r=find(coveredRows);
c=find(coveredCols);
[i,j]=find(A(r,c)<=0);
i=r(i);
j=c(j);
for k=1:length(i)
    lj=find(A(i(k),:)==-j(k));
    A(i(k),lj)=A(i(k),j(k));
    A(i(k),j(k))=0;
end
A(r,c)=A(r,c)+m;
end