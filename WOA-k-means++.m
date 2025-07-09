clc;
clear all;

% 示例数据
dataTable = readtable('特征数据.xls');
data = table2array(dataTable);
dim = size(data, 2); % 维度

% 设置聚类数目
K = 4;

% WOA参数
pop = 30; % 种群数量
Max_iter = 100; % 最大迭代次数
lb = repmat(min(data), K, 1); % 下边界
ub = repmat(max(data), K, 1); % 上边界

% 初始化最佳质心和成本
Best_pos = zeros(K, dim);
Best_Cost = inf;

% 定义目标函数
fobj = @(centroids)kmeans_cost(data, centroids, K);

% 调用WOA算法优化K-means++
[Best_Cost, Best_pos, curve, final_positions, final_costs, best_cluster_centers, best_cluster_assignments] = WOA(pop, Max_iter, lb, ub, dim, K, fobj, data);

% 输出优化后的参数
disp('优化后的质心：');
disp(Best_pos);
disp('最小成本：');
disp(Best_Cost);

% 绘制成本变化曲线
figure;
plot(curve, 'LineWidth', 2);
title('成本变化曲线');
xlabel('迭代次数');
ylabel('成本');
grid on;

% 使用优化后的质心进行K-means++聚类
opts = statset('Display', 'final');
[idx, centroids] = kmeans(data, K, 'Start', Best_pos, 'Options', opts);

% 将簇编号映射为颜色
colors = hsv(K);

% 创建图
figure;
hold on;
% 绘制每个类别的数据点
for i = 1:K
    clusterPoints = data(idx == i, :);
    scatter(clusterPoints(:, 1), clusterPoints(:, 2), 50, colors(i, :), 'filled');
end

% 绘制质心
plot(centroids(:, 1), centroids(:, 2), 'kx', 'MarkerSize', 10, 'LineWidth', 2);

% 设置图例
legend([arrayfun(@(x) ['类别' num2str(x)], 1:K, 'UniformOutput', false), {'质心'}], 'Location', 'eastoutside');

title('WOA-K-Means++ 聚类');
xlabel('特征1');
ylabel('特征2');
hold off;

% 绘制簇分配变化曲线
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

% WOA算法
function [Best_Cost, Best_pos, curve, final_positions, final_costs, best_cluster_centers, best_cluster_assignments] = WOA(pop, Max_iter, lb, ub, dim, K, fobj, data)

    % 初始化位置向量和领导者的得分
    Best_pos = zeros(K, dim);
    Best_Cost = inf; % 对于最小化问题
    
    % 初始化搜索代理的位置
    Positions = initialization(pop, K, dim, ub, lb);

    curve = zeros(1, Max_iter);
    t = 0; % 循环计数器
    final_costs = zeros(pop, 1); % 保存每个个体的最终成本
    best_cluster_centers = zeros(K, dim, Max_iter); % 保存每次迭代的最佳聚类中心
    best_cluster_assignments = zeros(size(data, 1), Max_iter); % 保存每次迭代的最佳聚类分配

    % 主循环
    while t < Max_iter
        for i = 1:pop
            % 返回超出搜索空间边界的搜索代理
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
            
            % 计算每个搜索代理的目标函数值
            fitness = fobj(squeeze(Positions(i, :, :)));
            final_costs(i) = fitness; % 保存每个个体的最终成本
            
            % 更新领导者
            if fitness < Best_Cost % 对于最大化问题，改为>
                Best_Cost = fitness; % 更新alpha
                Best_pos = squeeze(Positions(i, :, :));
                % 保存最佳聚类中心和分配
                [best_cluster_assignments(:, t+1), best_cluster_centers(:, :, t+1)] = kmeans(data, K, 'Start', Best_pos, 'MaxIter', 100, 'OnlinePhase', 'off');
            end
        end
        
        a = 2 - t * (2 / Max_iter); % a从2线性减少到0
        a2 = -1 + t * (-1 / Max_iter); % a2从-1线性减少到-2
        
        % 更新搜索代理的位置
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
    
    % 返回最终的所有个体位置
    final_positions = Positions;
end

% 初始化搜索代理的位置
function Positions = initialization(pop, K, dim, ub, lb)
    Positions = zeros(pop, K, dim);
    for i = 1:K
        for d = 1:dim
            Positions(:, i, d) = rand(pop, 1) .* (ub(i, d) - lb(i, d)) + lb(i, d);
        end
    end
end

% 计算K-means++的成本函数
function cost = kmeans_cost(data, centroids, K)
    [idx, ~] = kmeans(data, K, 'Start', centroids, 'MaxIter', 100, 'OnlinePhase', 'off');
    cost = sum(min(pdist2(data, centroids).^2, [], 2));
end
