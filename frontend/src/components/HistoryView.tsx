import { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useAuth } from '@/contexts/AuthContext';
import { api, VitalsResponse, ConversationResponse } from '@/lib/api';
import { useToast } from '@/hooks/use-toast';
import { Activity, MessageCircle, Loader2 } from 'lucide-react';

const HistoryView = () => {
  const { token } = useAuth();
  const { toast } = useToast();
  const [vitalsHistory, setVitalsHistory] = useState<VitalsResponse[]>([]);
  const [chatHistory, setChatHistory] = useState<ConversationResponse[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    if (!token) return;

    try {
      setIsLoading(true);
      const [vitals, chats] = await Promise.all([
        api.getVitalsHistory(token, 10),
        api.getConversationsHistory(token, 10),
      ]);
      setVitalsHistory(vitals);
      setChatHistory(chats);
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to load history',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };

  const getRiskColor = (risk: string) => {
    const riskLower = risk.toLowerCase();
    if (riskLower.includes('high')) return 'bg-destructive text-destructive-foreground';
    if (riskLower.includes('medium') || riskLower.includes('moderate')) return 'bg-warning text-warning-foreground';
    return 'bg-success text-success-foreground';
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const formatText = (text: string) => {
    const parts = text.split('**');
    return parts.map((part, index) => {
      if (index % 2 === 1) {
        return <strong key={index}>{part}</strong>;
      }
      return part;
    });
  };

  return (
    <Card className="max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle>Your Health History</CardTitle>
        <CardDescription>View your past vitals submissions and AI conversations</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="vitals">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="vitals" className="flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Vitals History
            </TabsTrigger>
            <TabsTrigger value="conversations" className="flex items-center gap-2">
              <MessageCircle className="h-4 w-4" />
              Conversations
            </TabsTrigger>
          </TabsList>

          <TabsContent value="vitals" className="mt-6">
            {isLoading ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
              </div>
            ) : vitalsHistory.length === 0 ? (
              <div className="text-center py-12 text-muted-foreground">
                No vitals history yet. Submit your first vitals assessment!
              </div>
            ) : (
              <ScrollArea className="h-[500px] pr-4">
                <div className="space-y-4">
                  {vitalsHistory.map((record) => (
                    <Card key={record.id}>
                      <CardHeader>
                        <div className="flex items-start justify-between">
                          <div>
                            <CardTitle className="text-base">
                              {formatDate(record.created_at)}
                            </CardTitle>
                            <CardDescription className="text-xs mt-1">
                              Vitals Record #{record.id}
                            </CardDescription>
                          </div>
                          <Badge className={getRiskColor(record.ml_risk_label)}>
                            {record.ml_risk_label}
                          </Badge>
                        </div>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-sm">
                          <div>
                            <span className="text-muted-foreground">Age:</span>
                            <span className="ml-2 font-medium">{record.age} years</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">BP:</span>
                            <span className="ml-2 font-medium">{record.systolic_bp}/{record.diastolic_bp}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">HR:</span>
                            <span className="ml-2 font-medium">{record.heart_rate} bpm</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">BS:</span>
                            <span className="ml-2 font-medium">{record.bs} mmol/L</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Temp:</span>
                            <span className="ml-2 font-medium">{record.body_temp}°{record.body_temp_unit === 'celsius' ? 'C' : 'F'}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Risk:</span>
                            <span className="ml-2 font-medium">{(record.ml_probability * 100).toFixed(1)}%</span>
                          </div>
                        </div>
                        {record.patient_history && (
                          <div className="pt-2 border-t">
                            <p className="text-xs text-muted-foreground mb-1">Patient History:</p>
                            <p className="text-sm">{record.patient_history}</p>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </ScrollArea>
            )}
          </TabsContent>

          <TabsContent value="conversations" className="mt-6">
            {isLoading ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
              </div>
            ) : chatHistory.length === 0 ? (
              <div className="text-center py-12 text-muted-foreground">
                No conversation history yet. Start chatting with the AI!
              </div>
            ) : (
              <ScrollArea className="h-[500px] pr-4">
                <div className="space-y-4">
                  {chatHistory.map((conversation) => (
                    <Card key={conversation.id}>
                      <CardHeader>
                        <CardTitle className="text-base">
                          {formatDate(conversation.created_at)}
                        </CardTitle>
                        <CardDescription className="text-xs">
                          Conversation #{conversation.id}
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        <div className="bg-muted/50 rounded-lg p-3">
                          <p className="text-xs text-muted-foreground mb-1">Your Question:</p>
                          <p className="text-sm font-medium">{conversation.user_message}</p>
                        </div>
                        <div className="bg-primary/5 rounded-lg p-3">
                          <p className="text-xs text-muted-foreground mb-1">AI Response:</p>
                          <p className="text-sm whitespace-pre-wrap leading-relaxed">
                            {formatText(conversation.ai_response)}
                          </p>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </ScrollArea>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default HistoryView;
