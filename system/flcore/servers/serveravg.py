import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        # Ratio of clients per round / 客户端总数
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        # Budget初始化为空list
        self.Budget = []


    def train(self):
        for i in range(self.global_rounds+1): # 全局轮数[0,self.global_round]
            # 当前round的开始时间
            s_t = time.time() 
            # 获得选中的客户端，是一个list，记录客户端的序号              
            self.selected_clients = self.select_clients()
            # server发送模型
            self.send_models()
            # 如果到达了评估模型的时间，默认为1，则开始评估客户端
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
            # 所有被选中的客户端进行 本地训练
            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            # server接受模型
            self.receive_models()

            # 暂时不知道？
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            # server聚合参数
            self.aggregate_parameters()

            # Budget记录当前轮数的运行时间
            self.Budget.append(time.time() - s_t)
            # 输出最近一轮的花费时间
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])
            # 如果可以自动退出 且 已完成
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        # 迭代完全局轮数后，输出
        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        # 计算没round得平均时间
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
        # 保存全局模型
        self.save_results()
        self.save_global_model()

        # 如果new_client存在，则进行评估
        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
