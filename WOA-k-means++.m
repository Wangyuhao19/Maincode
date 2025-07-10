clc;
clear all;

dataTable = readtable('Data.xls');
data = table2array(dataTable);
dim = size(data, 2); 


K = 4;

% WOA
pop = 30;
Max_iter = 100; 
lb = repmat(min(data), K, 1);
ub = repmat(max(data), K, 1);

Best_pos = zeros(K, dim);
Best_Cost = inf;


fobj = @(centroids)kmeans_cost(data, centroids, K);


[Best_Cost, Best_pos, curve, final_positions, final_costs, best_cluster_centers, best_cluster_assignments] = WOA(pop, Max_iter, lb, ub, dim, K, fobj, data);


disp('优化后的质心：');
disp(Best_pos);
disp('最小成本：');
disp(Best_Cost);


figure;
plot(curve, 'LineWidth', 2);
title('成本变化曲线');
xlabel('迭代次数');
ylabel('成本');
grid on;


opts = statset('Display', 'final');
[idx, centroids] = kmeans(data, K, 'Start', Best_pos, 'Options', opts);
colors = hsv(K);


figure;
hold on;
for i = 1:K
    clusterPoints = data(idx == i, :);
    scatter(clusterPoints(:, 1), clusterPoints(:, 2), 50, colors(i, :), 'filled');
end

plot(centroids(:, 1), centroids(:, 2), 'kx', 'MarkerSize', 10, 'LineWidth', 2);

legend([arrayfun(@(x) ['类别' num2str(x)], 1:K, 'UniformOutput', false), {'质心'}], 'Location', 'eastoutside');

title('WOA-K-Means++ 聚类');
xlabel('特征1');
ylabel('特征2');
hold off;

figure;
hold on;
for i = 1:size(data, 1)
    plot(1:Max_iter, best_cluster_assignments(i, :), 'LineWidth', 1.5);
end
title('簇分配变化曲线');
xlabel('迭代次数');
ylabel('簇编号');
grid on;
hold off;

function [Best_Cost, Best_pos, curve, final_positions, final_costs, best_cluster_centers, best_cluster_assignments] = WOA(pop, Max_iter, lb, ub, dim, K, fobj, data)


    Best_pos = zeros(K, dim);
    Best_Cost = inf; 
    Positions = initialization(pop, K, dim, ub, lb);

    curve = zeros(1, Max_iter);
    t = 0; 
    final_costs = zeros(pop, 1); 
    best_cluster_centers = zeros(K, dim, Max_iter);
    best_cluster_assignments = zeros(size(data, 1), Max_iter);


    while t < Max_iter
        for i = 1:pop
           
            Flag4ub = squeeze(Positions(i, :, :)) > ub;
            Flag4lb = squeeze(Positions(i, :, :)) < lb;
            for j = 1:K
                for d = 1:dim
                    if Flag4ub(j, d)
                        Positions(i, j, d) = ub(j, d);
                    elseif Flag4lb(j, d)
                        Positions(i, j, d) = lb(j, d);
                    end
                end
            end
            

            fitness = fobj(squeeze(Positions(i, :, :)));
            final_costs(i) = fitness; 
            

            if fitness < Best_Cost
                Best_Cost = fitness;
                Best_pos = squeeze(Positions(i, :, :));
                [best_cluster_assignments(:, t+1), best_cluster_centers(:, :, t+1)] = kmeans(data, K, 'Start', Best_pos, 'MaxIter', 100, 'OnlinePhase', 'off');
            end
        end
        
        a = 2 - t * (2 / Max_iter); 
        a2 = -1 + t * (-1 / Max_iter); 
        
        for i = 1:pop
            r1 = rand();
            r2 = rand();
            
            A = 2 * a * r1 - a;
            C = 2 * r2;
            
            b = 1;
            l = (a2 - 1) * rand + 1;
            
            p = rand();
            
            for j = 1:K
                for d = 1:dim
                    if p < 0.5   
                        if abs(A) >= 1
                            rand_leader_index = floor(pop * rand() + 1);
                            X_rand = squeeze(Positions(rand_leader_index, :, :));
                            D_X_rand = abs(C * X_rand(j, d) - Positions(i, j, d));
                            Positions(i, j, d) = X_rand(j, d) - A * D_X_rand;
                        elseif abs(A) < 1
                            D_Leader = abs(C * Best_pos(j, d) - Positions(i, j, d));
                            Positions(i, j, d) = Best_pos(j, d) - A * D_Leader;
                        end
                    elseif p >= 0.5
                        distance2Leader = abs(Best_pos(j, d) - Positions(i, j, d));
                        Positions(i, j, d) = distance2Leader * exp(b * l) * cos(l * 2 * pi) + Best_pos(j, d);
                    end
                end
            end
        end
        t = t + 1;
        curve(t) = Best_Cost;
    end
    
    final_positions = Positions;
end

function Positions = initialization(pop, K, dim, ub, lb)
    Positions = zeros(pop, K, dim);
    for i = 1:K
        for d = 1:dim
            Positions(:, i, d) = rand(pop, 1) .* (ub(i, d) - lb(i, d)) + lb(i, d);
        end
    end
end

function cost = kmeans_cost(data, centroids, K)
    [idx, ~] = kmeans(data, K, 'Start', centroids, 'MaxIter', 100, 'OnlinePhase', 'off');
    cost = sum(min(pdist2(data, centroids).^2, [], 2));
end
