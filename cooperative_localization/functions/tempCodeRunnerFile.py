      # シミュレーション全体におけるMSE及びRMSEの算出
      squared_error_total += np.sum(squared_error_list)
      # squared_error_total += np.nansum(squared_error_list)
      targets_localized_count_total += len(targets_localized)
      mean_squared_error = squared_error_total/targets_localized_count_total
      root_mean_squared_error = np.sqrt(mean_squared_error)
      
      # 求めたRMSEをリストに追加
      # root_mean_squared_error_list = np.append(root_mean_squared_error_list, root_mean_squared_error)

      # RMSE（シミュレーション平均）の算出
      if sim_cycle == 0:
        root_mean_squared_error_avg = root_mean_squared_error
      else:
        root_mean_squared_error_avg = (root_mean_squared_error_avg*sim_cycle + root_mean_squared_error)/(sim_cycle + 1)
      
      # RMSEの分布を更新（協調測位の場合はRMSEの値が大きく振れるのであまり意味がないかも）
      # field_rmse_distribution = rmse_distribution.update(field_rmse_distribution, grid_interval, targets_localized, squared_error_list)

      # 測位順と測位誤差のリスト
      # squared_error_lists = np.append(squared_error_lists, np.array([squared_error_list]), axis=0)

      # 測位可能確率の分布の更新とその平均の算出
      field_localizable_probability_distribution = localizable_probability_distribution.update(field_localizable_probability_distribution, grid_interval, targets, targets_localized)
      localizable_probability_avg = np.sum(field_localizable_probability_distribution[:, 2]*field_localizable_probability_distribution[:, 3])/np.sum(field_localizable_probability_distribution[:, 3])
      
      sim_cycle += 1
      positive = np.sum(features_list[:, -1] >= error_threshold)
      negative_weight5 = np.sum((features_list[:, -1] < (error_threshold - error_threshold_grid*4)) & (features_list[:, -1] > (error_threshold - error_threshold_grid*5)))
      negative_weight4 = np.sum((features_list[:, -1] < (error_threshold - error_threshold_grid*3)) & (features_list[:, -1] > (error_threshold - error_threshold_grid*4)))
      negative_weight3 = np.sum((features_list[:, -1] < (error_threshold - error_threshold_grid*2)) & (features_list[:, -1] > (error_threshold - error_threshold_grid*3)))
      negative_weight2 = np.sum((features_list[:, -1] < (error_threshold - error_threshold_grid*1)) & (features_list[:, -1] > (error_threshold - error_threshold_grid*2)))
      negative_weight1 = np.sum((features_list[:, -1] < (error_threshold )) & (features_list[:, -1] > (error_threshold - error_threshold_grid*1)))

      # positive = np.sum(features_list[:, -1] >= error_threshold)
      # negative = np.sum(features_list[:, -1] < error_threshold)  
      #   
      print(f"positive: {positive}/{sample_data_count} negative_w5: {negative_weight5}/{sample_data_count} negative_w4: {negative_weight4}/{sample_data_count} negative_w3: {negative_weight3}/{sample_data_count} negative_w2: {negative_weight2}/{sample_data_count} negative_w1: {negative_weight1}/{sample_data_count}", end=" ")
      # print(f"positive: {positive}/{sample_data_count} negative: {negative}/{sample_data_count}", end=" ")
      print("RMSE: " + "{:.4f}".format(root_mean_squared_error_avg) + " / Avg. Localizable Prob.: " + "{:.4f}".format(localizable_probability_avg), end="\r\r")
