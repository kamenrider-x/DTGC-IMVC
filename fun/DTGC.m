function [result,Clus,affinity2]=test6(X,V,N,W,lambda1,alpha,beta,gt)

epson = 1e-7;
K = size(unique(gt),1);
pr = 0.6;
sX = [N, N, V];
mu = 0.1;
mu_max = 10e10;
eta1 = 2;
eta2 = 2;
converge_Z=[]; converge_Z_G=[]; converge_Z_C=[]; converge_S_C=[];
Isconverg = 0;
iter = 1;

for v = 1:V
    mv = size(X{v}, 2); 
    H{v} = zeros(mv, mv);
    P{v} = zeros(N, N);
    U{v} = zeros(N, N);
    Z{v} = zeros(N, N);
    D{v} = zeros(N, N);
    J{v} = zeros(N, N);
    G{v} = zeros(N, N);
    S{v} = zeros(N, N);
    E1{v} = zeros(N, N);
    E2{v} = zeros(N, N);
    Y1{v} = zeros(N, N); 
    Y2{v} = zeros(N, N);
    Y3{v} = zeros(N, N);
    Y4{v} = zeros(N, N);
end
I = eye(N);

%  iteration
while(Isconverg == 0)

    % update J{v}
    J_tensor = cat(3, J{:,:});
    Y2_tensor = cat(3, Y2{:,:});
    j = J_tensor(:);
    y2 = Y2_tensor(:);
    [j, ~] = wshrinkObj(j+1/mu*y2, 1/mu, sX, 0, 3);
    J_tensor = reshape(j, sX);
    for v=1:V
        J{v} = J_tensor(:,:,v);
    end

    % update G{v}
    G_tensor = cat(3, G{:,:});
    Y4_tensor = cat(3, Y4{:,:});
    g = G_tensor(:);
    y4 = Y4_tensor(:);
    [g, ~] = wshrinkObj_weight_lp(g+1/mu*y4, alpha/mu, sX, 0, 3, pr);
    G_tensor = reshape(g, sX);
    for v=1:V
        G{v} = G_tensor(:,:,v);
    end
    
    % update E
    for v=1:V
        Q1 = U{v} - U{v}*(Z{v}+D{v}) + Y1{v}./mu;
        Q2 = Z{v} - S{v} + Y3{v}./mu;
        Q_concat = [Q1; Q2];
        
        E_concat = solve_l1l2(Q_concat, lambda1./mu);
        
        E1{v} = E_concat(1:N, :);
        E2{v} = E_concat(N+1:end, :);
    end

    % update H{v}
    for v= 1:V
        M_mat = I - Z{v} - D{v};
        C1 = E1{v} - Y1{v}./mu;
        
        A_sylv = 2 .* (X{v}' * X{v});
        B_sylv = mu .* (W{v} * M_mat * M_mat' * W{v}');
        C_sylv = 2 .* (X{v}' * X{v}) + mu .* (W{v} * (C1 - P{v} * M_mat) * M_mat' * W{v}');
        H{v} = sylvester(full(A_sylv), full(B_sylv), full(C_sylv));
    end

    % update P{v}
    for v= 1:V
        M_mat = I - Z{v} - D{v};
        C1 = E1{v} - Y1{v}./mu;
        F_mat = C1 - W{v}' * H{v} * W{v} * M_mat;
        O_idx = find(sum(W{v}, 1) > 0); 
        U_idx = find(sum(W{v}, 1) == 0); 
        
        MM = M_mat * M_mat' + epson * I;
        FM = F_mat * M_mat';
        P{v} = zeros(N, N); 
        if ~isempty(U_idx) && ~isempty(O_idx)
            P{v}(O_idx, U_idx) = FM(O_idx, U_idx) / MM(U_idx, U_idx);
        end
        if ~isempty(U_idx) && ~isempty(O_idx)
            P{v}(U_idx, O_idx) = P{v}(O_idx, U_idx)';
        end
    end

    % update U{v}
    for v= 1:V
        U{v} = P{v} + W{v}' * H{v} * W{v};
    end

    % update Z{v}
    for v= 1:V
        A1 = U{v} - U{v}*D{v} - E1{v} + Y1{v}./mu;
        A2 = J{v} - Y2{v}./mu;
        A3 = S{v} + E2{v} - Y3{v}./mu;
        
        Z{v} = inv(U{v}' * U{v} + 2 .* I) * (U{v}' * A1 + A2 + A3);
    end

    % update S{v}
    for v = 1:V
        K_mat = 0.5 .* (Z{v} - E2{v} + Y3{v}./mu + G{v} - Y4{v}./mu);
        % Projection onto S*1 = 1
        S{v} = K_mat + (1/N) .* (ones(N,1) - K_mat * ones(N,1)) * ones(1,N);
    end
 
    % update D{v}
    for v=1:V
        B1 = U{v} - U{v}*Z{v} - E1{v} + Y1{v}./mu;
        D{v} = inv(U{v}' * U{v} + (2*beta/mu) .* I) * (U{v}' * B1);
    end

    % update Y1{v}
    for v=1:V
        Y1{v} = Y1{v} + mu * (U{v} - U{v}*(Z{v}+D{v}) - E1{v});
    end

    % update Y2{v}
    for v=1:V
        Y2{v} = Y2{v} + mu * (Z{v} - J{v});
    end

    % update Y3{v}
    for v=1:V
        Y3{v} = Y3{v} + mu * (Z{v} - S{v} - E2{v});
    end

    % update Y4{v}
    for v=1:V
        Y4{v} = Y4{v} + mu * (S{v} - G{v});
    end

    %check
    max_Z=0;
    max_Z_G=0;
    max_Z_C=0;
    max_S_C=0;
    Isconverg = 1;
    for c = 1:V
        if (norm(U{c} - U{c}*(Z{c}+D{c}) - E1{c}, inf) > epson)
            history.norm_Z = norm(U{c} - U{c}*(Z{c}+D{c}) - E1{c}, inf);
            Isconverg = 0;
            max_Z = max(max_Z, history.norm_Z);
        end
        if (norm(Z{c} - J{c}, inf) > epson)
            history.norm_Z_G = norm(Z{c} - J{c}, inf);
            Isconverg = 0;
            max_Z_G = max(max_Z_G, history.norm_Z_G);
        end

        if (norm(Z{c} - S{c}-E2{c}, inf) > epson)
            history.norm_Z_C = norm(Z{c} - S{c}-E2{c}, inf);
            Isconverg = 0;
            max_Z_C = max(max_Z_C, history.norm_Z_C);
        end

        if (norm(S{c} - G{c}, inf) > epson)
            history.norm_S_C = norm(S{c} - G{c}, inf);
            Isconverg = 0;
            max_S_C = max(max_S_C, history.norm_S_C);
        end
    end
    
    converge_Z=[converge_Z max_Z];
    converge_Z_G=[converge_Z_G max_Z_G];
    converge_Z_C=[converge_Z_C max_Z_C];
    converge_S_C=[converge_S_C max_S_C];
    
    % update mu
    mu  = min(mu_max,mu*eta2);
    iter = iter + 1;
    if (iter==100)
        Isconverg = 1;
    end
end

affinity1 = zeros(N,N);
affinity2 = zeros(N,N);
for v = 1:V
    affinity1 = affinity1+abs(S{v})+abs(S{v}');
    affinity2 = affinity2+abs(Z{v})+abs(Z{v}');
end
affinity1 = affinity1/V;
affinity2 = affinity2/V;

Clus = Ng_SpectralClustering(affinity1,K);

[ACC,NMI,PUR] = ClusteringMeasure(gt,Clus); %ACC NMI Purity
[Fscore,Precision,R] = compute_f(gt,Clus);
[AR,~,~,~]=RandIndex(gt,Clus);
result = [ACC NMI PUR Fscore Precision R AR];
%  figure
%  plot(converge_S_C, '-b.', 'LineWidth', 2,'MarkerSize', 10);
% h=gca;
% set(h,'FontSize', 18); % 设置字体大小为12
%  xlabel('iteration',FontSize=18);
%  ylabel('Error',FontSize=18);
%  legend('Caltech7',FontSize=18);
end