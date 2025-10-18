import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useAuth } from '@/contexts/AuthContext';
import VitalsForm from '@/components/VitalsForm';
import RiskAssessment from '@/components/RiskAssessment';
import ChatInterface from '@/components/ChatInterface';
import HistoryView from '@/components/HistoryView';
import type { VitalsSubmitResponse } from '@/lib/api';
import { LogOut, Activity, MessageCircle, History } from 'lucide-react';
import logo from '@/assets/logo.png';

const Dashboard = () => {
  const { username, logout } = useAuth();
  const navigate = useNavigate();
  const [latestAssessment, setLatestAssessment] = useState<VitalsSubmitResponse | null>(null);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const handleVitalsSubmit = (data: VitalsSubmitResponse) => {
    setLatestAssessment(data);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary/5 via-background to-secondary/10">
      <header className="border-b bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <img src={logo} alt="AfyaJamii" className="h-12 w-12" />
            <div>
              <h1 className="text-xl font-bold text-foreground">AfyaJamii</h1>
              <p className="text-sm text-muted-foreground">Maternal Health Platform</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-right">
              <p className="text-sm font-medium">Welcome, {username}</p>
              <p className="text-xs text-muted-foreground">Your health, our priority</p>
            </div>
            <Button variant="outline" size="sm" onClick={handleLogout}>
              <LogOut className="h-4 w-4 mr-2" />
              Logout
            </Button>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <Tabs defaultValue="vitals" className="space-y-6">
          <TabsList className="grid w-full grid-cols-3 max-w-2xl mx-auto">
            <TabsTrigger value="vitals" className="flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Vitals
            </TabsTrigger>
            <TabsTrigger value="chat" className="flex items-center gap-2">
              <MessageCircle className="h-4 w-4" />
              AI Chat
            </TabsTrigger>
            <TabsTrigger value="history" className="flex items-center gap-2">
              <History className="h-4 w-4" />
              History
            </TabsTrigger>
          </TabsList>

          <TabsContent value="vitals" className="space-y-6">
            <VitalsForm onSubmitSuccess={handleVitalsSubmit} />
            {latestAssessment && <RiskAssessment data={latestAssessment} />}
          </TabsContent>

          <TabsContent value="chat">
            <ChatInterface />
          </TabsContent>

          <TabsContent value="history">
            <HistoryView />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
};

export default Dashboard;
